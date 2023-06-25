# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        if self.total() != 0:
            floatOfTotal = float(self.total())
            for key in self.keys():
                self[key] /= floatOfTotal

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        totalSUM= self.total()


        if totalSUM != 1: # check if the sum of the values is equal to 1, if not then normalize
            self.normalize()
        sortedITEMS = sorted(self.items()) # sort the items in the dictionary according to keys
        distribution = []
        val = []
        nRandom = random.random() # generates a random float number between 0 and 1
        x = 0

        for itm in sortedITEMS:
            distribution.append(itm[1]) # append the value of each item to distribution list
        
        for itm in sortedITEMS:
            val.append(itm[0]) # append the key of each item to val list
        
        distTtal = distribution[0]
        
        if nRandom > distTtal: # checks if the randomly generated number is greater than the first item in distribution list
            while nRandom > distTtal: # increment x and add the next value of distribution until nRandom is less than distTtal
                x = x + 1
                distTtal = distTtal + distribution[x]
        else: 
            return val[x] # return the key of the first item in the sorted list if nRandom is not greater than distTtal
        
        newval=val[x] # get the key of the xth item in the sorted list
        return newval # return the key of the item that nRandom is greater than its value```


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
     # Set variables to the respective parameters that were passed in
        noNoise=None
        noDistance=None
        jailPOS= jailPosition
        ghostPOS= ghostPosition
        pacPOS= pacmanPosition
        nullNoise=0
        positiveNoise=1

        # If the ghost is in jail and there is no noise, return a probability of 1. 
        # If there is noise, return a probability of 0.
        if ghostPOS is jailPOS:
            if noisyDistance is noNoise:
                return positiveNoise
            else:
                return nullNoise

        # If there is no noisy distance, return a probability of 0.
        # Calculate the Manhattan distance between the pacman position and ghost position.
        if noisyDistance is noNoise:
            return nullNoise
        dist = manhattanDistance(pacPOS, ghostPOS)

        # If there is no distance between the pacman position and ghost position, return a probability of 0.
        if dist is noDistance:
            return nullNoise

        # Calculate the probability of observing the given noisy distance, given the calculated distance and return it.
        noisy = busters.getObservationProbability(
            noisyDistance, dist
        )
        return noisy

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        passedObservation= observation # Observation passed

        # For each position in self.allPositions:
        for currentPOS in self.allPositions:
            # Get the jail position and pacman position
            positionJW = self.getJailPosition()
            positionPAC = gameState.getPacmanPosition()
            
            # Calculate the observation probability for the current position
            observationProb = self.getObservationProb(passedObservation, positionPAC, currentPOS, positionJW)
            
            # If the observation probability is False, return the normalized beliefs
            if observationProb is False:
                return self.beliefs.normalize()
            else:
                # Update the belief for the current position
                belliefOfCurrent = self.beliefs[currentPOS]
                self.beliefs[currentPOS] = observationProb * belliefOfCurrent

        # Normalize the beliefs and return them
        self.beliefs.normalize()


    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # create a new discrete distribution object
        discDist = DiscreteDistribution()
        # loop through each possible ghost position
        for current in self.allPositions:#^
            # get the distribution over new positions given the current position
            newPosDist = self.getPositionDistribution(gameState, current)#^
            # loop through each possible new position and compute its weighted probability
            itms=newPosDist.items()
            for (updatedpac, prob) in itms:#^
                # update the weighted probability for the current new position
                discDist[updatedpac] = discDist[updatedpac] + ( self.beliefs[current] * prob) #^
        # update the belief distribution with the new discrete distribution
        self.beliefs = discDist
    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        numOFparticals=range(self.numParticles)
        # Calculate the number of legal positions
        countLegalPositions=len(self.legalPositions)

        # Calculate the number of particles to place in each legal position
        pInPosition = round(self.numParticles / countLegalPositions, 2)

        # For each legal position:
        for pacmanPOS in self.legalPositions:
            if pacmanPOS is False:
                return 0
            else:
                # Add the appropriate number of particles to the particles list for that position
                for x in numOFparticals:
                    self.particles.append(pacmanPOS)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # get the jail position and Pacman position
        positionJW = self.getJailPosition()
        positionPAC = gameState.getPacmanPosition()
        # create a new DiscreteDistribution object
        discDist = DiscreteDistribution()
        # create a copy of the current particles and initialize a count
        selfparticles = self.particles.copy()
        count = range(self.numParticles)
        null=0

        # loop through each particle and update its weight in discDist
        for esc in selfparticles:
            observationProb = self.getObservationProb(observation, positionPAC, esc, positionJW)
            discDist[esc] = discDist[esc] + observationProb
        discDist.normalize() # normalize the distribution

        # check if the total probability is 0
        totalD = discDist.total()
        if totalD == null:
            # if the total probability is 0, reset the belief state
            self.initializeUniformly(gameState)
            discDist = self.getBeliefDistribution()
            selfparticles = self.particles.copy()
            self.particles = [] # create a new set of particles by sampling from discDist
            for x in count:
                self.particles.append(discDist.sample())
        else:
            self.particles = [] # create a new set of particles by sampling from discDist
            for x in count:
                self.particles.append(discDist.sample())


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # Get the current belief distribution
        observationProb = self.getBeliefDistribution()

        # Create a new DiscreteDistribution object
        discDist = DiscreteDistribution()
        #Num of particals
        numParticals = range(self.numParticles)
        # For each legal position:
        for currentpos in self.legalPositions:
            # Get the position distribution for the current position
            newPosDist = self.getPositionDistribution(gameState, currentpos)
            itms=newPosDist.items()
            if newPosDist is 0:
                return 0
            else:
                # For each possible next position and its probability in the position distribution:
                for nextPOS, prob in itms:
                    # Update the discrete distribution for the next position with the probability of being at the current position times the probability of transitioning to the next position
                    discDist[nextPOS] = discDist[nextPOS] + (prob * observationProb[currentpos])

        # Clear the current particles list
        self.particles = []

        # For each particle to be placed:
        for particle in numParticals:
            # Sample a new particle from the updated discrete distribution and add it to the particles list
            self.particles.append(discDist.sample())


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        discDist = DiscreteDistribution()
        paricales=self.particles
        null=0
        # loop through each particle and increment the weight of the corresponding position
        for particle in paricales:
            discDist[particle] = discDist[particle] + 2.50
        # normalize the distribution
        if discDist==1:
            return null
        else:
            discDist.normalize()
            # return the belief distribution
            return discDist


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            raiseNotDefined()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
