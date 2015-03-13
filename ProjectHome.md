Using emergent programing techniques on the GPU we have made an implementation of sugarscape to utilize the massively parallel architecture of modern GPUs.  Agents within the model move optimally within their vision which is uniformly set between [1,10]. Multiple agent cannot occupy the same cell. The agents also interact with the sugar patches uniformly given a metabolism between [0.1,1).  The sugar patches grow at a constant rate of 0.1 per time step until they reach their maximum values which are determined by two Gaussian functions.