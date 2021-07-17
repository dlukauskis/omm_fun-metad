from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from simtk.openmm.app.metadynamics import *
from sys import stdout
import os
import subprocess as sp
import shutil

import numpy as np
import MDAnalysis as mda
from pathlib import Path



"""
OpenMM-native implementation of funnel-metadynamics
Steps:
    1. 10k enmin steps
    2. 5ns solute-restrained NPT equilibration, with MC bstat
    3. 5ns Calpha+ligand-restrained NVT equilibration
    4. 2000ns well-tempered funnel-metadynamics

Outputs:
    1. COLVAR, logging CVs every 2 ps
    2. bias files, written every 10 ns
    3. trajectory, written every 1000 ps
    4. checkpoint, every 1000 ps
"""

run_time = 2000.0 # ns

if os.path.isfile('checkpnt.chk'):
    print('Loading a restart')
    if os.path.isfile('sim_log.csv'):
        os.rename('sim_log.csv','0_sim_log.csv')
    sim_log_file = [ line[:-2] for line in open('0_sim_log.csv').readlines()]
    current_steps = int(sim_log_file[-1].split(',')[1])
    steps = run_time * 250000 - current_steps
    is_restart = True
else:
    steps = run_time * 250000
    is_restart = False

def get_CA_indices(coords, parm):

    if coords.endswith('.gro'):
        u = mda.Universe(coords)
    elif coords.endswith('.rst') or coords.endswith('.rst7') or coords.endswith('.inpcrd'):

        u = mda.Universe(parm, coords, format='INPCRD')

    ca_atoms = u.select_atoms('name CA')

    ca_list = [int(ca) for ca in ca_atoms.indices]

    return ca_list

def get_ligand_indices(coords, parm, lig_name = 'MOL', include_h = True):

    if coords.endswith('.gro'):
        u = mda.Universe(coords)
    elif coords.endswith('.rst') or coords.endswith('.rst7') or coords.endswith('.inpcrd'):

        u = mda.Universe(parm, coords, format='INPCRD')

    if include_h is False:
        lig_atoms = u.select_atoms('resname %s and not (name h* or name H*)'% lig_name)
    else:
        lig_atoms = u.select_atoms('resname %s'% lig_name)

    lig_list = [int(a) for a in lig_atoms.indices]

    return lig_list

def get_protein_indices(coords, parm, include_h = True):

    if coords.endswith('.gro'):
        u = mda.Universe(coords)
    elif coords.endswith('.rst') or coords.endswith('.rst7') or coords.endswith('.inpcrd'):

        u = mda.Universe(parm, coords, format='INPCRD')

    if include_h is False:
        prot_atoms = u.select_atoms('protein and not name H*')
    else:
        prot_atoms = u.select_atoms('protein')

    prot_list = [int(a) for a in prot_atoms.indices]

    return prot_list

def run_10k_enmin(params, input_positions):

    # prepare system
    system = params.createSystem(nonbondedMethod=PME,
                               nonbondedCutoff=1.0*nanometers,
                               constraints=HBonds,
                               rigidWater=True,
                               ewaldErrorTolerance=0.0005)

    integrator = LangevinIntegrator(300*kelvin, 1.0/picoseconds,
                                      2.0*femtoseconds)
    integrator.setConstraintTolerance(0.00001)

    # prepare simulation
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaPrecision':'mixed'}
    simulation = Simulation(params.topology, system,
                            integrator, platform, properties)

    simulation.context.setPositions(input_positions)

    ### 10k min, with 10kJ target ###
    simulation.minimizeEnergy(maxIterations=10000, tolerance=10*kilojoule/mole)

    p = simulation.context.getState(getPositions=True).getPositions()

    return p

def run_5ns_NPT_restaint_equil(params, input_positions):

    ### 5ns in NPT (MonteCarlo bstat) with 5kcal all heavy solute atom restraints ###
    # prepare system
    system = params.createSystem(nonbondedMethod=PME,
                               nonbondedCutoff=1.0*nanometers,
                               constraints=HBonds,
                               rigidWater=True,
                               ewaldErrorTolerance=0.0005)

    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)

    system.addForce(restraint)
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    atomsToRestrain = get_protein_indices(coords_file, parm_file, include_h = False) + get_ligand_indices(coords_file, parm_file, include_h = False)

    dummyIndex = []
    positions = input_positions
    for i in atomsToRestrain:
        j = system.addParticle(0)
        nonbonded.addParticle(0, 1, 0)
        nonbonded.addException(i, j, 0, 1, 0)
        restraint.addBond(i, j, 0*nanometers, 5*kilocalories_per_mole/angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])
    integrator = LangevinIntegrator(300*kelvin, 1.0/picoseconds,
                                      2.0*femtoseconds)
    system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
    context = Context(system, integrator)

    context.setPositions(input_positions)

    run_time = 5.0 # ns

    print('Initial energy:', context.getState(getEnergy=True).getPotentialEnergy())
    integrator.step(int(run_time * 500000))
    print('Final energy:', context.getState(getEnergy=True).getPotentialEnergy())

    p = context.getState(getPositions=True).getPositions()[:dummyIndex[0]]

    return p

def run_5ns_NVT_restaint_equil(params, input_positions):

    ### 5ns in NVT with 5kcal for Ca and ligand heavy atoms ###

    # prepare system
    system = params.createSystem(nonbondedMethod=PME,
                                 nonbondedCutoff=1.0*nanometers,
                                 constraints=HBonds,
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)

    restraint = HarmonicBondForce()
    restraint.setUsesPeriodicBoundaryConditions(True)

    system.addForce(restraint)
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    atomsToRestrain = get_CA_indices(coords_file, parm_file) + get_ligand_indices(coords_file, parm_file, include_h = False)

    dummyIndex = []
    positions = input_positions
    for i in atomsToRestrain:
        j = system.addParticle(0)
        nonbonded.addParticle(0, 1, 0)
        nonbonded.addException(i, j, 0, 1, 0)
        restraint.addBond(i, j, 0*nanometers, 5*kilocalories_per_mole/angstrom**2)
        dummyIndex.append(j)
        input_positions.append(positions[i])

    integrator = LangevinIntegrator(300*kelvin, 1.0/picoseconds,
                                      2.0*femtoseconds)
    context = Context(system, integrator)

    context.setPositions(input_positions)

    run_time = 5.0 # ns

    print('Initial energy:', context.getState(getEnergy=True).getPotentialEnergy())
    integrator.step(int(run_time * 500000))
    print('Final energy:', context.getState(getEnergy=True).getPotentialEnergy())

    p = context.getState(getPositions=True).getPositions()[:dummyIndex[0]]

    PDBFile.writeFile(params.topology, p, open('equilibrated.pdb', 'w'))

    return p

"""The actual start of the script"""

coords_file = '../../input/2XHT/solvated.rst7'
parm_file = '../../input/2XHT/solvatedHMR.prmtop'

coords = AmberInpcrdFile(coords_file)
parm = AmberPrmtopFile(parm_file)


# run the minimisation and equilibration sims
if not os.path.isfile('equilibrated.pdb'):
    print('Starting energy minimization')
    min_pos = run_10k_enmin(parm, coords.positions)
    print('Done')
    print('Starting restrained NPT equilibration')
    npt_pos = run_5ns_NPT_restaint_equil(parm, min_pos)
    print('Done')
    print('Starting restrained NVT equilibration')
    input_pos = run_5ns_NVT_restaint_equil(parm, npt_pos)
    print('Done')
else:
    input_pos = PDBFile('equilibrated.pdb').getPositions()

print('Starting production metadynamics simulation')
# and now the metadynamics production run
# prepare system and integrator
system = parm.createSystem(nonbondedMethod=PME,
                           nonbondedCutoff=1.0*nanometers,
                           constraints=HBonds,
                           rigidWater=True,
                           ewaldErrorTolerance=0.0005)



lig = [3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395]
p1 = [1267, 1281, 1300, 2143, 2159, 2633, 2647, 2669, 2685]
p2 = [581, 596, 615, 634, 645, 659, 670, 681, 693, 703, 1316, 1342, 1349, 1368, 1375, 1438, 1450, 1502, 1516, 1948, 2143, 2178, 2626, 2633, 2647, 2669]

projection = CustomCentroidBondForce(3, 'distance(g1,g2)*cos(angle(g1,g2,g3))')
projection.addGroup(lig)
projection.addGroup(p1)
projection.addGroup(p2)
projection.addBond([0,1,2])
projection.setUsesPeriodicBoundaryConditions(True)

proj = BiasVariable(projection, 0.00, 3.70, 0.025, False, gridWidth = 500)

alignment_indices = lig + p1

rmsd = RMSDForce(input_pos, alignment_indices)

grid_min, grid_max = 0.0, 2.0 # nm
hill_width = 0.02 # nm

rmsd_cv = BiasVariable(rmsd, grid_min, grid_max, hill_width, False, gridWidth = 500)

# add a flat-bottom restraint
k1 = 10000*kilojoules_per_mole
k2 = 1000*kilojoules_per_mole
upper_wall = 3.50*nanometer
lower_wall = 0.20


upper_wall_rest = CustomCentroidBondForce(3, '(k/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - upper_wall, 0)^2')
upper_wall_rest.addGroup(lig)
upper_wall_rest.addGroup(p1)
upper_wall_rest.addGroup(p2)
upper_wall_rest.addBond([0,1,2])
upper_wall_rest.addGlobalParameter('k', k1)
upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
# dont forget to actually add the force to the system ;)
system.addForce(upper_wall_rest)

# this is the restraint for the sides of the cone/funnel
# Ill give a full derivation a bit later in a ipynb
wall_width = 1.00*nanometer
beta_cent = 1.50
s_cent = 1.50*nanometer
wall_buffer = 0.15*nanometer

dist_restraint = CustomCentroidBondForce(3, '(k/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a/(1+exp(b*(distance(g1,g2)*cos(angle(g1,g2,g3))-c)))+d), 0)^2')
dist_restraint.addGroup(lig)
dist_restraint.addGroup(p1)
dist_restraint.addGroup(p2)
dist_restraint.addBond([0,1,2])
dist_restraint.addGlobalParameter('k', k2)
dist_restraint.addGlobalParameter('a', wall_width)
dist_restraint.addGlobalParameter('b', beta_cent)
dist_restraint.addGlobalParameter('c', s_cent)
dist_restraint.addGlobalParameter('d', wall_buffer)
dist_restraint.setUsesPeriodicBoundaryConditions(True)
system.addForce(dist_restraint)

# now the bottom of the funnel
if lower_wall == 0:
    #lower_wall = 0.01 didnt work, singularities
    lower_wall = 0.05 # close enough to zero...
lower_wall = lower_wall*nanometer
lower_wall_rest = CustomCentroidBondForce(3, '(k/2)*min(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall, 0)^2')
lower_wall_rest.addGroup(lig)
lower_wall_rest.addGroup(p1)
lower_wall_rest.addGroup(p2)
lower_wall_rest.addBond([0,1,2])
lower_wall_rest.addGlobalParameter('k', k1)
lower_wall_rest.addGlobalParameter('lower_wall', lower_wall)
lower_wall_rest.setUsesPeriodicBoundaryConditions(True)
system.addForce(lower_wall_rest)


### Set up the simulation. ###

if not os.path.isdir('bias_dir'):
    os.mkdir('bias_dir')

# print bias every 10ns
meta = Metadynamics(system, [proj,rmsd_cv], 300.0*kelvin, 10.0, 1.5*kilojoules_per_mole, 500, biasDir = 'bias_dir/', saveFrequency = 2500000)

integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.004*picoseconds)
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision':'mixed'}

simulation = Simulation(parm.topology, system, integrator, platform, properties)
simulation.context.setPositions(input_pos)

simulation.minimizeEnergy()

if os.path.isfile('checkpnt.chk'):
    simulation.loadCheckpoint('checkpnt.chk')

simulation.reporters.append(DCDReporter('trj.dcd', 250000, append=is_restart)) # every 1ns
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 250000)) # every 1ns
simulation.reporters.append(StateDataReporter(
                           'sim_log.csv', 2500000, step=True,
                           temperature=True,progress=True,
                           remainingTime=True,speed=True,
                           totalSteps=steps,separator=',')) # print every 10ns
if is_restart: 
    current_cvs = meta.getCollectiveVariables(simulation)
    colvar_array = np.load('COLVAR.npy')
    colvar_array = np.append(colvar_array, [current_cvs], axis=0)
else:
    colvar_array = np.array([meta.getCollectiveVariables(simulation)])

for i in range(0,int(steps),500):
    if i%10000 == 0:
        np.save('COLVAR.npy', colvar_array)
    if i%(2500000/2) == 0 and i != 0:
        paths = sorted(Path('bias_dir/').iterdir(), key=os.path.getmtime)
        bias_files_by_age = [ str(fpath) for fpath in paths if 'bias' in str(fpath) and not '_bias' in str(fpath)]
        last_bias_file = bias_files_by_age[-1]
        sp.call('cp %s bias_dir/_%s'% (last_bias_file,last_bias_file.split('/')[-1]), shell=True)
    meta.step(simulation, 500)
    current_cvs = meta.getCollectiveVariables(simulation)
    colvar_array = np.append(colvar_array, [current_cvs], axis=0)
print('Done')

