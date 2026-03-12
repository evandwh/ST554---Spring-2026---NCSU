import numpy as np
import matplotlib.pyplot as plt

class SLR_slope_simulator:

    def __init__(self, beta_0, beta_1, x, sigma, seed):

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.sigma = sigma
        self.x = np.array(x)
        self.n = len(x)
        self.rng = np.random.default_rng(seed)
        self.slopes = []

    def generate_data(self):
        """
        Generate one dataset based on the SLR model:
        y = beta_0 + beta_1*x + epsilon
        Data set will returned in a list containing two arrays, one for each variable.
        """
        epsilon = self.rng.normal(0, self.sigma, self.n)
        y = self.beta_0 + self.beta_1 * self.x + epsilon
        return self.x, y

    def fit_slope(self, x, y):
        """
        Fit an SLR and return the estimated slope (beta_1_hat)
        """
        
        x_bar = np.mean(x)
        y_bar = np.mean(y)

        top = np.sum((x - x_bar) * (y - y_bar))
        bottom = np.sum((x - x_bar)**2)

        beta_1_hat = top / bottom
        return beta_1_hat


    def run_simulations(self, num_simulations):
        """
        Run multiple simulations and store slope estimates in an array.
        """

        local_slopes = []
        for _ in range(num_simulations):
            x, y = self.generate_data()
            slope_hat = self.fit_slope(x, y)
            local_slopes.append(slope_hat)
            
        self.slopes = np.array(local_slopes)

    def plot_sampling_distribution(self):
        if len(self.slopes) == 0:
            print("You must call run_simulations() first!")
        else:
            plt.hist(self.slopes, bins = 30)
            plt.xlabel("Slope Estimates")
            plt.ylabel("Frequency")
            plt.title("Approx. Sampling Distribution of Slope Estimates")
            plt.show()

    def find_prob(self, value, sided):
        if len(self.slopes) == 0:
            print("You must run run_simulations() first.")
            return

        n = len(self.slopes)
        
        if sided == "above":
            prob = np.sum(self.slopes > value) / n
            
        elif sided == "below":
            prob = np.sum(self.slopes < value) / n


        elif sided == "two-sided":
            median = np.median(self.slopes)

            if value > median:
                prob = 2 * (np.sum(self.slopes > value) / n)
            else:
                prob = 2 * (np.sum(self.slopes < value) / n)

        else:
            print("You did not give correct value for sided. Must be either 'above', 'below', or 'two-sided'.")
            return

        return prob


#Testing the code out using the instructions from class.

sim = SLR_slope_simulator(beta_0=12, beta_1=2,
                          x=np.array(list(np.linspace(start=0, stop=10, num=11)) * 3),
                          sigma=1, seed=10
)

#Testing to see if method gives the correct error.
sim.plot_sampling_distribution()

# Running 10000 simulations
sim.run_simulations(10000)

# Plotting our slope estimates
sim.plot_sampling_distribution()

# Approximating the probability it is more extreme that 2.1
prob = sim.find_prob(2.1, "two-sided")
print("Two-sided probability:", prob)

# Print simulated slopes
print(sim.slopes)