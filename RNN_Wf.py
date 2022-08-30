from gettext import translation
import tensorflow as tf
import numpy as np
import os

class RNNWaveFunction:
  """
  Inputs:
  - memory_units: length of state vector
  - activation_function:
  - N_sites: number of spins in the chain
  - N_samples: number of samples for training or measuring
  - J: array of magnetic couplings 
  """
  def __init__(self,n_layers,memory_units,activation_function,N_sites,N_samples,J):
    self.n_layers = n_layers
    self.memory_units = memory_units
    self.activation_function = activation_function
    self.states0 = [tf.zeros(shape=(1,self.memory_units)) for _ in range(self.n_layers)]
    self.N_sites = N_sites
    self.N_samples = N_samples 
    self.J = J 
    self.get_multilayer_rnn() # initializes an RNN

  def pi_soft_sign(self,inputs):
    return np.pi * tf.nn.softsign(inputs)
  
  def heaviside(self,integer):
    """ 
    Home-made heaviside function using tf.sign()
    Note: tf.sign(0) = 0
    """
    x = tf.sign(tf.cast(tf.sign(integer),dtype=tf.float32) - 0.1 ) 
    return 0.5*(tf.cast(tf.sign(x),dtype=tf.float32)+1.0)

  def get_multilayer_rnn(self):
    """
    This method defines self.rnn as an object with 4 layers.
    - layer 0: Input layer (not used...)
    - layer 1: Multilayer RNN cell, whose output h is the hidden state
    - layer 2: Fully-connected layer that gives the absolute value of conditional
      probabilities of each spin
    - layer 3: Fully-connected layer that gives the phase of each spin
    """
    inputs = tf.keras.Input(shape=(1,2),tensor=tf.zeros(shape=(1,2)), name = 'Input_layer')

    rnn_cells = [tf.keras.layers.GRUCell(units=self.memory_units,
                 activation=self.activation_function) for _ in range(self.n_layers)]

    h, _ = tf.keras.layers.StackedRNNCells(rnn_cells)(inputs,states=self.states0)

    dense_prob = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax,
                                           name = 'Softmax_layer')(h)

    dense_phase = tf.keras.layers.Dense(units=2, activation=self.pi_soft_sign,
                                        name = 'Softsign_layer')(h)

    self.rnn = tf.keras.Model(inputs=inputs, outputs=[dense_prob, dense_phase])

    return

  def load_weights(self, modelsfolder, L):
    """
    Loads the weights saved in 'modelsfolder' of a 
    trained model corresponding to an L-site system
    """

    self.rnn.load_weights(modelsfolder + 'L' + str(L) + '/weights.h5')
    return

  @tf.function
  def sample_rnn(self):
    """
    Returns: 
    - system_sample: (self.N_sites,)
    - conditionals: (self.N_sites,2)
    """

    conditionals = tf.TensorArray(tf.float32, size=self.N_sites,name='conditionals')
    phases = tf.TensorArray(tf.float32, size=self.N_sites,name='phases')
    system_sample = tf.TensorArray(tf.int64, size=self.N_sites,name='sample')

    # Initially, the recurrent cell is fed with null vectors
    rnn_output, rnn_state = self.rnn.layers[1](
                            inputs=tf.zeros(shape=(1,2)),
                            states=self.states0)
    # Conditional probabilities  1
    conditional = self.rnn.layers[2](rnn_output)
    conditionals = conditionals.write(0, conditional)
    # Phase 1
    phase = self.rnn.layers[3](rnn_output)[0]
    phases = phases.write(0,phase)
    # First spin
    spin_sample = tf.random.categorical(tf.math.log(conditional), 1)
    system_sample = system_sample.write(0, spin_sample[0][0])
    # Iterate
    for i in tf.range(1, self.N_sites):
        rnn_output, rnn_state = self.rnn.layers[1](
                                tf.one_hot(spin_sample[0], 2),
                                rnn_state)
        conditional = self.rnn.layers[2](rnn_output)
        phase = self.rnn.layers[3](rnn_output)[0]
        if False: # not yet finished
          if(i + 1 > int(self.N_sites / 2)):
            conditional = self.fixing_magnetization_sector(i,system_sample.stack(),conditional)
        spin_sample = tf.random.categorical(
                      tf.math.log(conditional), 1)
        phases = phases.write(i,phase)
        conditionals = conditionals.write(i, conditional)
        system_sample = system_sample.write(i, spin_sample[0][0])

    return system_sample.stack(), conditionals.stack(), phases.stack()
  
  @tf.function
  def sample_probability_and_phase(self, sample, conditionals, phases):
    """
    Inputs: 
    - sample: (self.N_sites,) a sample drawn from the RNN
    - conditionals: (self.N_sites,1,2) the conditional probabilities obtained during sampling
    - phases: (self.N_sites,2) the phases obtained during sampling

    Returns:
    - sample probability: real positive number between 0 and 1
    - sample_phase: real number between -pi and pi 
    """

    sample_conditionals = tf.TensorArray(tf.float32, size=tf.shape(sample)[0])
    sample_phase = tf.zeros(shape=())
    for i in tf.range(tf.shape(sample)[0]):
        sample_conditionals = sample_conditionals.write(i,
                              tf.tensordot(conditionals[i][0],
                              tf.one_hot(sample[i], 2), axes=[[0], [0]]))
        sample_phase += tf.tensordot(phases[i],tf.one_hot(sample[i], 2), axes=[[0], [0]])
    return tf.math.reduce_prod(sample_conditionals.stack()), sample_phase

  @tf.function
  def draw_samples(self):
    """
    This function draws 'self.N_samples' of length 
    'self.N_sites' using 'self.rnn'

    Returns:
    - samples: (self.N_samples,self.N_sites)
    - probs: (self.N_samples,)
    - phases: (self.N_samples,)
    """
    samples = tf.TensorArray(tf.int64, size=self.N_samples,
                             element_shape=(self.N_sites,))
    samples_probs = tf.TensorArray(tf.float32, size=self.N_samples,
                                   element_shape=())
    samples_phases = tf.TensorArray(tf.float32, size=self.N_samples,
                                    element_shape=())
    
    for i in tf.range(self.N_samples):
      system_sample, conditionals, phases = self.sample_rnn()
      probability, phase = self.sample_probability_and_phase(system_sample, conditionals, phases) 
      samples = samples.write(i, system_sample)
      samples_probs = samples_probs.write(i, probability)
      samples_phases = samples_phases.write(i, phase)

    return samples.stack(), samples_probs.stack(), samples_phases.stack()

  @tf.function
  def spin_fluctuation(self, sample, position):
    """
    This method takes a sample and creates a spin fluctuation
    between sites "position" and "position+1", exchanging the corresponding spin projections.

    Inputs:
    - sample: (self.N_sites,)
    - position: integer between 0 and "self.N_sites-2"

    Returns:
    - state: (self.N_sites,) a state generated from "sample" by fliping the spins in
    "position" and "position+1"
    """
    state = tf.TensorArray(dtype = tf.int64, size = self.N_sites,
                           element_shape=())

    for i in tf.range(position):
      state = state.write(i, sample[i])
    
    state = state.write(position, sample[position + 1])
    state = state.write(position+1, sample[position])

    if (position + 2 <= self.N_sites - 1):
      for i in tf.range(position+2,self.N_sites):
        state = state.write(i, sample[i])

    return state.stack()

  @tf.function
  def H_matrix_elements(self, sample):
    """
    This function takes a "sample" and computes the finite matrix elements
    of the Hamiltonian: H_{sigma,sigma'}.
    Each sigma' is a state generated by fliping a pair of neighboring spins in "sample".

    Inputs:
    - smaple (self.N_sites,)

    Returns:
    - diag: H_{sigma,sigma}, the diagonal element for state sigma
    - matrix_elements: (n,) a list of matrix_elements H_{sigma,sigma'},
      with sigma'!=sigma. n is sample-dependent, and corresponds to the
      number of states sigma'.
    - states: (n,self.N_sites) a list of states sigma'
    """
    matrix_elements = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True,
                                     element_shape=())
    states = tf.TensorArray(dtype = tf.int64, size = 0, dynamic_size = True,
                            element_shape=(self.N_sites,)) 

    # Diagonal term
    diag = tf.zeros(shape=(), dtype = tf.float32) 
    for site in tf.range(self.N_sites - 1):
      diag += self.J[site] * (0.25) * \
              tf.math.cos(tf.cast(sample[site] + sample[site+1],dtype=tf.float32) * tf.constant(np.pi))

    # Off-diagonal terms
    counter = tf.zeros(shape=(),dtype = tf.int32) 
    for site in tf.range(self.N_sites - 1):
      if (sample[site] != sample[site+1]):
        states = states.write(counter, self.spin_fluctuation(sample, site))
        matrix_elements = matrix_elements.write(counter, self.J[site] / 2.)
        counter += 1

    return diag, matrix_elements.stack(), states.stack()

  @tf.function
  def get_state_prob_and_phase(self, state):
    """
    Inputs:
    - state: (self.N_sites)

    Returns:
    - state probability
    - state phase
    """
    state_conditionals = tf.TensorArray(tf.float32, size=self.N_sites, name='conditionals',
                                         element_shape=())
    state_phases = tf.TensorArray(tf.float32, size=self.N_sites, name='phases',
                                   element_shape=())

    # Initially, the recurrent cell is fed with null vectors
    rnn_output, rnn_state = self.rnn.layers[1](
                            inputs=tf.zeros(shape=(1,2)),
                            states=self.states0
                            )
    # Probability of sigma_1
    conditionals = self.rnn.layers[2](rnn_output)[0]
    state_conditionals = state_conditionals.write(0,
                          tf.tensordot(conditionals, tf.one_hot(state[0], 2), axes=[[0], [0]])
                          )
    # Phase 1
    phases = self.rnn.layers[3](rnn_output)[0]
    state_phases = state_phases.write(0,
                    tf.tensordot(phases, tf.one_hot(state[0], 2), axes=[[0], [0]])
                    )
    # Iterate
    for i in tf.range(1, self.N_sites):
      rnn_output, rnn_state = self.rnn.layers[1](
                              tf.one_hot([state[i-1]], 2),
                              rnn_state)    
      conditionals = self.rnn.layers[2](rnn_output)[0]         
      state_conditionals = state_conditionals.write(i,
                            tf.tensordot(conditionals, tf.one_hot(state[i], 2), axes=[[0], [0]])
                            )
      phases = self.rnn.layers[3](rnn_output)[0]
      state_phases = state_phases.write(i,
                      tf.tensordot(phases, tf.one_hot(state[i], 2), axes=[[0], [0]])
                      )
    return tf.math.reduce_prod(state_conditionals.stack()), tf.math.reduce_sum(state_phases.stack())

  @tf.function
  def E_loc(self, sample, sample_prob, sample_phase):
    """
    Inputs:
    - sample: (self.N_sites)
    - sample_prob
    - sample_phase

    Returns:
    - Eloc: the local energy for the given input sample
    """
    diag, matrix_elements, states = self.H_matrix_elements(sample)
    n_states = tf.shape(matrix_elements)[0]
    matrix_elements = tf.complex(matrix_elements,tf.zeros(shape=(n_states,)))
    Eloc = tf.complex(diag,0.)

    for i in tf.range(n_states):
      state_prob, state_phase = self.get_state_prob_and_phase(states[i])
      state_psi = tf.complex(tf.math.sqrt(state_prob),0.) * tf.cast(tf.math.exp(tf.complex(0.,state_phase)),tf.complex64)
      sample_psi = tf.complex(tf.math.sqrt(sample_prob),0.) * tf.cast(tf.math.exp(tf.complex(0.,sample_phase)),tf.complex64)
      Eloc += matrix_elements[i] * state_psi / sample_psi

    return Eloc
  
  @tf.function
  def local_energies(self, samples, probs, phases):
    """
    Inputs:
    - samples: (self.N_samples, self.N_sites)
    - probs: (self.N_samples,)
    - phases: (self.N_samples,)

    Returns:
    - Eloc: (self.N_samples,) a tensor with the local energy for each sample 
    """
    Eloc=tf.TensorArray(size=self.N_samples, dtype=tf.complex64)
    for i in tf.range(self.N_samples):
      Eloc = Eloc.write(i, self.E_loc(samples[i], probs[i], phases[i]))

    return Eloc.stack()

  @tf.function
  def train_step(self, optimizer):
    """
    The train step consists in:
    - Draw a set of 'self.N_samples' samples from the RNN
    - Compute an estimation of the energy
    - Compute gradients using automatic differentiation (AD)
    - Update the network trainable parameters

    Returns:
    - Energy estimation over the given set of samples

    For AD bibliography: 

    - https://www.tensorflow.org/guide/autodiff
    - https://github.com/acevedo-s/AD_example
    - https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

    To understand Eloc and cost, see paper "Recurrent neural network wave functions"
    """
    with tf.GradientTape() as tape:
      samples, probs, phases = self.draw_samples()
      Eloc = self.local_energies(samples, probs, phases)
      cost = 2 * tf.math.real(
             tf.reduce_mean(tf.multiply(
             tf.cast(tf.dtypes.complex((1/2.)*tf.math.log(probs), -phases), dtype=tf.complex64), tf.stop_gradient(Eloc)))
             -tf.reduce_mean(tf.cast(tf.dtypes.complex((1/2.)*tf.math.log(probs), -phases), dtype=tf.complex64)) * tf.reduce_mean(tf.stop_gradient(Eloc))
             )
      grads = tape.gradient(cost, self.rnn.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.rnn.trainable_weights))
    tf.print('lr: ', optimizer._decayed_lr('float32'))
    return tf.reduce_mean(Eloc)

  def training(self, n_steps, lr, flags, modelsfolder, N_sites_load):
    """
    Inputs:
    - n_steps: number of training steps
    - lr: learning rate
    - flags: external dictionary with flags to choose if one wants to save and/or load models
    - modelsfolder: path to save the trained model
    - N_sites_load: number of sites of the system with which the loaded model was trained

    Outputs: 
    - Energies: (n_steps,) A list with the convergence 
      of E to the ground state estimated value 
    """
    # Load previous model
    if flags['load_weights']:
      print(f'loading model trained with {N_sites_load} sites\n')
      self.load_weights(modelsfolder, N_sites_load)

    # Preparation
    print('starting training:')
    Energies = []
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    # Training
    if flags['save_weights']:
      os.makedirs(modelsfolder + 'L' + str(self.N_sites) + '/', exist_ok = True)
      for step in range(n_steps):
        E = self.train_step(optimizer)
        Energies.append((E/self.N_sites).numpy())
        print(f'iteration {step + 1} | mean e: {E / self.N_sites:.6f}')
        if ((step+1)%10 == 0):
          self.rnn.save_weights(modelsfolder + 'L' + str(self.N_sites) + '/weights.h5')
    else:
      for step in range(n_steps):
        E = self.train_step(optimizer)
        Energies.append((E/self.N_sites).numpy())
        print(f'iteration {step + 1} | mean e: {E / self.N_sites:.6f}')
    
    if flags['save_history']:
      os.makedirs(modelsfolder + 'L' + str(self.N_sites) + '/', exist_ok = True)
      np.save(modelsfolder + 'L' + str(self.N_sites) + '/history', np.array(Energies))

    return Energies

  def measure_e_m(self):
    """
    Inputs:
    - 

    Note: 'self.draw_samples' uses 'self.N_samples' and 'self.rnn'

    Returns:  measurements, a list of observables measured
    The measurement is  computed in a generated set of samples.
    The list contains:
    - average energy estimation per site (complex scalar with zero imaginary part)
    - associated standard deviation (real scalar)
    - (N_samples,) magnetizations of each generated sample
    """
    measurements = []
    samples, probs, phases = self.draw_samples()
    # Energy
    Eloc = self.local_energies(samples, probs, phases)
    mean_e = tf.reduce_mean(Eloc) / self.N_sites
    std_e = tf.math.reduce_std(Eloc) / self.N_sites
    measurements.append(mean_e)
    measurements.append(std_e)
    # Magnetization
    samples = 2*tf.cast(samples, dtype = tf.float32) - 1 # normalization
    magnetizations = tf.reduce_sum(samples, axis = 1) / self.N_sites
    measurements.append(magnetizations)

    return measurements

  

######################################## END OF CODE ########################################

  @tf.function
  def conditional_transformation(self, conditional, N_up, N_down):
    """
    This method applies the transformation from appendix D.2 in "RNN wave functions"
    on the conditional probabilities to generate samples with zero magnetization.

    Inputs:
    - conditional: (1,2) conditional probabilities for the next spin in the chain
    - N_up: the number of up spins so far in the chain
    - N_down: the number of down spins so far in the chain

    Returns:
    - the transformed conditionals
    """
    transformed_conditionals = tf.TensorArray(tf.float32, size=2,
                                              name='transformed_conditionals')
    transformed_conditionals = transformed_conditionals.write(0,
                               conditional[0][0] * self.heaviside(int(self.N_sites / 2) - N_down))
    transformed_conditionals = transformed_conditionals.write(1,
                               conditional[0][1] * self.heaviside(int(self.N_sites / 2) - N_up))
    return tf.stack([tf.nn.l2_normalize(transformed_conditionals.stack(), epsilon = 1e-30)])

  @tf.function
  def fixing_magnetization_sector(self, index, system_sample, conditional):
    """
    This method calculates the number of up and down spins in the chain and applies 
    the method "self.conditional_transformation" on the conditional probabilities
    in order to produce configurations with zero magnetization.

    Inputs:
    - index: integer that indexes the position of the next spin to generate
    - system_sample: (self.N_sites,) but with only "i" generated spin
      and completed with zeros.
    - conditional: (1,2) conditional probabilities for the next spin before the transformation
    """
    N_up = int(tf.reduce_sum(system_sample))
    N_down = index - N_up
    return self.conditional_transformation(conditional, N_up, N_down)


  def get_rnn(self):
    inputs = tf.keras.Input(shape=(1,2),tensor=tf.zeros(shape=(1,2)), name = 'Input_layer')

    x, _ = tf.keras.layers.GRUCell(units=self.memory_units,
                                   activation=self.activation_function, name = 'RNN_cell') \
                                   (inputs, states=tf.zeros(shape=(1,self.memory_units)))

    dense_prob = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax,
                                           name = 'Softmax_layer')(x)

    dense_phase = tf.keras.layers.Dense(units=2, activation=self.pi_soft_sign,
                                        name = 'Softsign_layer')(x)

    self.rnn = tf.keras.Model(inputs=inputs, outputs=[dense_prob, dense_phase])

    return