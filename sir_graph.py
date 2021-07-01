#Basic model of the SIR epidemiology curve,
#attempting to showcase the limit of recovered individuals
#based on the ratio of an arbitrary recovery rate to infection rate
#
#R(limit->infinity) = (r/c)log((1 - I0)/1-

from scipy.special import lambertw
import matplotlib.pyplot as plt
import numpy as np
from math import e
import sys

class SIR_Model:

    def __init__(self, beta, gamma, I0, S0, Rec0 = 0):
        """
        Initializes object with c = infection rate, r = recovery rate,
        I0 = the total population.
        """
        #Declare time increment for approximations
        self.delta_t = 0.001
        #Declare max length of x-axis for graph
        self.max_value_length = 100

        #Set constants
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.N = I0 + S0 + Rec0

        #Set pandemic initial values
        self.I0 = float(I0)
        self.S0 = float(S0)
        self.Rec0 = Rec0

        #Derive infamous R0 ratio
        self.R0 = beta / gamma

        #Display basic data
        self.print_init_data()
        print()
        self.calculate_pandemic_end()
        print()

        #Euler's algorithm implementation
        self.susceptible = [self.S0]
        self.infected = [self.I0]
        self.recovered = [self.Rec0]
        self.x_axis = [self.delta_t]

        self.euler_approx()

        #Print out results
        self.return_estimated_end_values()
        self.peak_num_infected()

        #Print Graph
        self.run_graph()

    def print_init_data(self):
        """
        Lists the initial values of the pandemic, as well as
        infection ratio R0.
        :return:
        """
        print(f"Initial population size: {self.N:20.0f}")
        print()
        print(f"Initial susceptible population: {self.S0:13.0f}")
        print(f"Initial number of infected: {self.I0:13.0f}")
        print(f"Initial number of recovered: {self.Rec0:12.0f}")
        print(f"R0: {self.R0:41.3f}")


    def get_S_infinity(self):
        """
        Derives S(limit -> infinity) based on initialization values
        Returns a float, with the imaginary aspect of the complex number
        returned by lambertw thrown out.
        """
        lambert_input = (-self.S0 / self.N * self.R0 *
                         e ** (-self.R0*(1-self.Rec0 / self.N)))
        lambert_result = lambertw(lambert_input)
        result = (-1) * lambert_result / self.R0
        return result.real

    def calculate_pandemic_end(self):
        """
        Calculates the limit of susceptible and recovered individuals
        as time reaches infinity.
        In other words, the number of susceptible and recovered
        individuals as the pandemic ends.

        This is calculated using the Lambert W function to solve
        for the s->infinity limit of S(t).

        Postcondition: Returns the calculated end value of
        recovered individuals as a float.
        :return:
        """
        s_infinite = self.N * self.get_S_infinity()
        r_infinite = self.N * (1 - self.get_S_infinity())

        print(f"Calculated susceptible individuals: "
              f"{s_infinite:12.3f}")
        print(f"Calculated recovered individuals: "
              f"{r_infinite:15.3f}")

        return r_infinite

    def euler_approx(self):
        """
        Modifies the susceptible, infected, and recovered
        list data members to reflect an entire range of
        a pandemic. Uses the data member delta_t to determine
        the level of specifity of the approximation.

        Note that this is not an exact numerical calculation!
        This is an approximation that depends on the number of
        previously decided time increments.
        :return:
        """

        Sn = self.susceptible[-1]
        In = self.infected[-1]
        Rn = self.recovered[-1]

        for x in range((int) (self.max_value_length /
                              self.delta_t)):
            Sn = self.susceptible[-1]
            In = self.infected[-1]
            Rn = self.recovered[-1]
            last_x = self.x_axis[-1]

            ds_dt = self.S_prime(Sn, In)
            dr_dt = self.R_prime(Rn, In)
            di_dt = self.I_prime(In, Sn)

            self.susceptible.append(Sn + ds_dt)
            self.recovered.append(Rn + dr_dt)
            self.infected.append(In + di_dt)
            self.x_axis.append(last_x + self.delta_t)

            #Pandemic ends when number of infected is equal
            #to zero!
            if ((int) (self.infected[-1]) ==
                0): break


    def S_prime(self, s_prev, i_prev):
        """
        Returns the rate of change of S with respect to time
        as a float, based on a previous value of S and I,
        and a time increment.

        s_prev = Any previous value of S
        i_prev = A previous value of I at the same time as s_prev
        time_change = The time increment, where the rate of
        change of S being returned is that of (time of s_prev
        + time_change).
        :return:
        """

        s_prime_prev = (-self.beta / self.N) * i_prev * s_prev

        return self.delta_t * s_prime_prev

    def R_prime(self, r_prev, i_prev):
        """
        Returns the rate of change of R with respect to time as
        a float, based on a previous value of I and a time
        increment.

        r_prev = a previous value of r from the imemdiately
        earlier time increment
        i_prev = a previous value of i from the immediately
        earlier time increment.
        :param i_prev:
        :return:
        """

        return self.delta_t * self.gamma * i_prev

    def I_prime(self, i_prev, s_prev):
        """
        Calculates the next value of infected individuals
        based on the SIR model differential equation:
        dI/dt = (beta * S(t) * I(t) / N) - gamma * I(t)

        NOTE: Returns result of derivative equation multiplied
        by the time increment! Must be added to the previous value
        to get the actual number of infected individuals.

        :param s_now: The number of susceptible individuals in this
        time increment.
        :param r_now: The number of recovered individuals in this time
        increment.
        :return:
        """

        di_dt = (1.0 * self.beta / self.N) \
                * i_prev * s_prev - (self.gamma * i_prev)


        return self.delta_t * di_dt

    def run_graph(self):
        """
        Plots the derived values of susceptible, infected,
        and recovered individals using matplotlib.

        Susceptible = Red
        Infected = Blue
        Recovered = Green
        :return:
        """

        plt.plot(self.x_axis, self.susceptible, 'r--', label="Susceptible")
        plt.plot(self.x_axis, self.infected, 'b--', label = "Infected")
        plt.plot(self.x_axis, self.recovered, 'g--', label = "Recovered")
        plt.legend(loc="best")
        plt.xlabel('Time (arbitrary)')
        plt.ylabel('Population')

    def return_estimated_end_values(self):
        """
        Prints the last values of the list data members of the
        object, which were derived using Euler's algorithm to
        approximate a differential equation.

        Precondition: This MUST be called AFTER self.euler_approx()
        is called. Otherwise it will simply return the starting
        values of the pandemic.
        :return:
        """

        print(f"Estimated susceptible individuals: "
              f"{self.susceptible[-1]:13.3f}")
        print(f"Estimated recovered individuals: "
              f"{self.recovered[-1]:16.3f}")

    def peak_num_infected(self):
        """
        Finds the highest number of infected throughout the
        entire pandemic.

        Precondition: Must call self.euler_approx() first, otherwise
        this will return the starting number of infected.
        Postcondition: Prints a string.
        :return:
        """

        max_val = self.infected[0]
        for x in range(len(self.infected)):
            if self.infected[x] > max_val:
                max_val = self.infected[x]

        print (f"Highest number of infected: {max_val:21.3f}")



if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print("Correct usage...\n\n"
              "sir_graph.py [infective rate]"
              "[recovery rate] [initial num infected] "
              "[initial susceptible population]")
        sys.exit(-1)

    beta = (float) (sys.argv[1])
    gamma = (float) (sys.argv[2])
    init_infect = (float) (sys.argv[3])
    init_suscept = (float) (sys.argv[4])

    model = SIR_Model(beta, gamma,
                      init_infect, init_suscept)