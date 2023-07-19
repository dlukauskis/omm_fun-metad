import os
import math
import numpy as np
from metadynamics import *  # patched with getHillHeight
try:
    from simtk.openmm import *
    from simtk.openmm.app import *
    from simtk import unit
#    from simtk.openmm.app.metadynamics import *
except ImportError or ModuleNotFoundError:
    from openmm import *
    from openmm.app import *
    from openmm import unit
#    from openmm.app.metadynamics import *
from parmed import load_file

def get_current_bias(bias, position, variables):
    """Takes 'position' tuple with CV values and 'variables' CV objects
    to extract the instantaneus bias from a numpy matrix called 'bias'
    """
    axisGaussians = []
    # for each collective variable...
    for i,v in enumerate(variables):
        # ...calculate the distance to the current position along the CV ...
        x = (position[i]-v.minValue) / (v.maxValue-v.minValue)
        dist = np.abs(np.linspace(0, 1.0, num=v.gridWidth) - x)
        # ... and draw a gaussian-shaped curve along it
        scaledVariance = (v.biasWidth/(v.maxValue-v.minValue))**2
        axisGaussians.append(np.exp(-0.5*dist*dist/scaledVariance))
    # find where the peak of the gaussian is along each CV
    peak_indices = ([ i.argmax() for i in axisGaussians ])
    # get the current bias at the present CV1 and CV2 values
    current_bias = bias[peak_indices[1]][peak_indices[0]]

    return current_bias

run_time = 20.0 # ns
total_steps = run_time * 500000

bias_every_n_steps = 1000  # 2 ps
print_every_n_steps = 1000  # 2 ps
metad_temp = 300
bias_factor = 10
deltaT = metad_temp*(bias_factor-1)

coords = load_file('input.gro')
parm = load_file('input.top')

# Transfer the unit cell information from the GRO file to the top object
parm.box = coords.box[:]
# prepare system and integrator
system = parm.createSystem(nonbondedMethod=NoCutoff,
                           constraints=HBonds)

lig = [147,148,149,150,151,152,153,154]
p1 = [7,28,49,70,91,111,133]
p2 = [4,25,46,67,88,109,130]

projection = CustomCentroidBondForce(3, 'distance(g1,g2)*cos(angle(g1,g2,g3))')
projection.addGroup(lig)
projection.addGroup(p1)
projection.addGroup(p2)
projection.addBond([0,1,2])
projection.setUsesPeriodicBoundaryConditions(True)
sigma_cv1 = 0.025
cv1 = BiasVariable(projection, -0.4, 2.2, sigma_cv1, False, gridWidth = 200)

extent = CustomCentroidBondForce(3, 'distance(g1,g2)*sin(angle(g1,g2,g3))')
extent.addGroup(lig)
extent.addGroup(p1)
extent.addGroup(p2)
extent.addBond([0,1,2])
extent.setUsesPeriodicBoundaryConditions(True)
sigma_cv2 = 0.03
cv2 = BiasVariable(extent, 0.0, 1.0, sigma_cv2, False, gridWidth = 200)

# add a flat-bottom restraint
u_spring_k = 10000*unit.kilojoules_per_mole
wall_spring_k = 10000*unit.kilojoules_per_mole
l_spring_k = 10000*unit.kilojoules_per_mole
upper_wall = 2.0*unit.nanometer
lower_wall = -0.2*unit.nanometer

upper_wall_rest = CustomCentroidBondForce(3, '(k)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - upper_wall, 0)^2')
upper_wall_rest.addGroup(lig)
upper_wall_rest.addGroup(p1)
upper_wall_rest.addGroup(p2)
upper_wall_rest.addBond([0,1,2])
upper_wall_rest.addGlobalParameter('k', u_spring_k)
upper_wall_rest.addGlobalParameter('upper_wall', upper_wall)
upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
uwall_f_idx = system.addForce(upper_wall_rest)
# need to set a forcegroup so you can refer to it later when retrieving energies/forces
upper_wall_rest.setForceGroup(uwall_f_idx)

# this is the restraint for the sides of the cone/funnel
# Ill give a full derivation a bit later in a ipynb
wall_width = 0.50*unit.nanometer
beta_cent = 2.0
s_cent = 1.0*unit.nanometer
wall_buffer = 0.15*unit.nanometer

wall_restraint = CustomCentroidBondForce(3, '(k)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a/(1+exp(b*(distance(g1,g2)*cos(angle(g1,g2,g3))-c)))+d), 0)^2')
wall_restraint.addGroup(lig)
wall_restraint.addGroup(p1)
wall_restraint.addGroup(p2)
wall_restraint.addBond([0,1,2])
wall_restraint.addGlobalParameter('k', wall_spring_k)
wall_restraint.addGlobalParameter('a', wall_width)
wall_restraint.addGlobalParameter('b', beta_cent)
wall_restraint.addGlobalParameter('c', s_cent)
wall_restraint.addGlobalParameter('d', wall_buffer)
wall_restraint.setUsesPeriodicBoundaryConditions(True)
wall_bias_f_idx = system.addForce(wall_restraint)
# need to set a forcegroup so you can refer to it later when retrieving energies/forces
wall_restraint.setForceGroup(wall_bias_f_idx)

# now the bottom of the funnel
if lower_wall == 0*unit.nanometer:
    #lower_wall = 0.01 didnt work, singularities
    lower_wall = 0.05*unit.nanometer # close enough to zero...
# lower_wall = lower_wall*nanometer
lower_wall_rest = CustomCentroidBondForce(3, '(k)*min(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall, 0)^2')
lower_wall_rest.addGroup(lig)
lower_wall_rest.addGroup(p1)
lower_wall_rest.addGroup(p2)
lower_wall_rest.addBond([0,1,2])
lower_wall_rest.addGlobalParameter('k', l_spring_k)
lower_wall_rest.addGlobalParameter('lower_wall', lower_wall)
lower_wall_rest.setUsesPeriodicBoundaryConditions(True)
lwall_f_idx = system.addForce(lower_wall_rest)
# need to set a forcegroup so you can refer to it later when retrieving energies/forces
lower_wall_rest.setForceGroup(lwall_f_idx)

meta = Metadynamics(system, [cv1, cv2], metad_temp*unit.kelvin, bias_factor,
                    1.5*unit.kilojoules_per_mole, bias_every_n_steps,
                    biasDir = '.', saveFrequency = 500000)

integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision':'mixed'}

simulation = Simulation(parm.topology, system, integrator, platform, properties)
simulation.context.setPositions(coords.positions)

simulation.minimizeEnergy()

# Add reporters.
simulation.reporters.append(DCDReporter('openmm.dcd', 50000))  # write frame every 0.1 ns
log_file = open('openmm.log', 'a')
simulation.reporters.append(StateDataReporter(log_file,
                                              50000,
                                              step=True,
                                              time=True,
                                              potentialEnergy=True,
                                              kineticEnergy=True,
                                              totalEnergy=True,
                                              temperature=True,
                                              volume=True,
                                              totalSteps=total_steps,
                                              speed=True,
                                              remainingTime=True,
                                              separator=' '))  # write log every 0.1 ns

is_restart = False

# Create PLUMED compatible HILLS file.
if is_restart:
    h_file = open('HILLS','a')
else:
    h_file = open('HILLS','w')
    h_file.write('#! FIELDS time pp.proj pp.ext sigma_pp.proj sigma_pp.ext height biasf\n')
    h_file.write('#! SET multivariate false\n')
    h_file.write('#! SET kerneltype gaussian\n')

# Create PLUMED compatible COLVAR file.
if is_restart:
    c_file = open('COLVAR','a')
else:
    c_file = open('COLVAR','w')
    c_file.write('#! FIELDS time pp.proj pp.ext s_cent beta_cent wall_width wall_buffer lwall.bias lwall.force2 uwall.bias uwall.force2 wall_center scaling spring wall_bias finalbias.bias finalbias.wall_bias_bias meta.bias meta.work\n')

# Initialise the collective variable array.
cv1_value, cv2_value = meta.getCollectiveVariables(simulation)
current_hill_height = meta.getHillHeight(simulation)

# Write the initial collective variable record.
time = 0
write_line = f'{time:15} {cv1_value:20.16f} {cv2_value:20.16f}          {sigma_cv1}           {sigma_cv2} {current_hill_height:20.16f}            {bias_factor}\n'
h_file.write(write_line)

# Run the simulation.
steps_per_cycle = bias_every_n_steps
total_cycles = math.ceil(total_steps/steps_per_cycle)
if is_restart:
    remaining_steps = total_steps - step
else:
    remaining_steps = total_steps
remaining_cycles = math.ceil(remaining_steps / steps_per_cycle)
start_cycles = total_cycles - remaining_cycles
checkpoint = 500
if is_restart:
    percent_complete = 100 * (step / total_steps)
    print('Loaded state from an existing simulation.')
    print(f'Simulation is {percent_complete}% complete.')

meta_work = 0
for x in range(start_cycles, total_cycles):
    meta.step(simulation, steps_per_cycle)
    cv1_value, cv2_value = meta.getCollectiveVariables(simulation)
    current_hill_height = meta.getHillHeight(simulation)
    time = int((x+1) * 0.002*steps_per_cycle)
    write_line = f'{time:15} {cv1_value:20.16f} {cv2_value:20.16f}          {sigma_cv1}           {sigma_cv2} {current_hill_height:20.16f}            {bias_factor}\n'
    h_file.write(write_line)

    # write down the bias potentials and force squares for the restraints
    lwall_state = simulation.context.getState(getForces=True,getEnergy=True,groups={lwall_f_idx})
    lwall_bias = lwall_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    lwall_force = lwall_state.getForces(asNumpy=True)
    lwall_force2 = np.sum(np.square(lwall_force))/2

    uwall_state = simulation.context.getState(getForces=True,getEnergy=True,groups={uwall_f_idx})
    uwall_bias = uwall_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    uwall_force = uwall_state.getForces(asNumpy=True)
    uwall_force2 = np.sum(np.square(uwall_force))/2

    wall_bias_state = simulation.context.getState(getForces=True,getEnergy=True,groups={wall_bias_f_idx})
    wall_bias = wall_bias_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    wall_bias_force = wall_bias_state.getForces(asNumpy=True)
    wall_bias_force2 = np.sum(np.square(wall_bias_force))/2

    #! FIELDS time pp.proj pp.ext rmsd rmsd_wall s_cent beta_cent wall_width wall_buffer lwall.bias lwall.force2 uwall.bias uwall.force2 rmsd_wall.bias rmsd_wall.force2 wall_center scaling spring wall_bias finalbias.bias finalbias.wall_bias_bias meta.bias
    # radius of the funnel at the CURRENT pp.proj value
    wall_center = wall_width.value_in_unit(unit.nanometer)*(1.0/(1.0+np.exp(beta_cent*(cv1_value-s_cent.value_in_unit(unit.nanometer)))))+wall_buffer.value_in_unit(unit.nanometer)
    scaling = 1.0
    finalbias_bias = wall_bias
    finalbias_wall_bias_bias = wall_bias
    fes = meta.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    bias = -fes/((metad_temp+deltaT)/deltaT)
    meta_bias = get_current_bias(bias, (cv1_value, cv2_value), [cv1, cv2])
    meta_work += current_hill_height/((metad_temp+deltaT)/deltaT)  # newer versions of PLUMED dont write this anymore
    c_line = f"{time:>8.6f} {cv1_value:>8.4f} {cv2_value:>8.4f} {s_cent.value_in_unit(unit.nanometer):>8.4f} {beta_cent:>8.4f} {wall_width.value_in_unit(unit.nanometer):>8.4f} {wall_buffer.value_in_unit(unit.nanometer):>8.4f} {lwall_bias:>8.4f} {lwall_force2:>8.4f} {uwall_bias:>8.4f} {uwall_force2:>8.4f} {wall_center:>8.4f} {scaling:>8.4f} {wall_spring_k.value_in_unit(unit.kilojoule_per_mole):>8.4f} {wall_bias:>8.4f} {finalbias_bias:>8.4f} {finalbias_wall_bias_bias:>8.4f} {meta_bias:>8.4f} {meta_work:>8.4f}\n"
    c_file.write(c_line)

    # write to the files more often, not just at the end
    # one 'cycle' is 2 ps, so save every 2 ns
    if x % 1000 == 0:
        c_file.close()
        h_file.close()
        c_file = open('COLVAR','a')
        h_file = open('HILLS','a')

h_file.close()
c_file.close()
