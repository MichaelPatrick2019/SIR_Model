#Basic model of the SIR epidemiology curve,
#attempting to showcase the limit of recovered individuals
#based on the ratio of an arbitrary recovery rate to infection rate
#
#R(limit->infinity) = (r/c)log((1 - I0)/1-

from scipy.special import lambertw
import matplotlib.pyplot as plt
import numpy as np
from math import e

class SIR_Model:

    def __init__(self, beta, gamma, I0, S0, Rec0 = 0):
        """
        Initializes object with c = infection rate, r = recovery rate,
        I0 = the total population.
        """
        #Declare time increment for approximations
        self.delta_t = 0.01
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

        #Euler's algorithm implementation
        self.susceptible = [self.S0]
        self.infected = [self.I0]
        self.recovered = [self.Rec0]
        self.x_axis = [self.delta_t]

        self.euler_approx()

        #Print out results
        self.return_estimated_end_values()
        self.peak_num_infected()

        """
        print("Susceptible list: ")
        print(self.susceptible)

        print("Infected list: ")
        print(self.infected)

        print("Recovered list: ")
        print(self.recovered)
        """

        #Print Graph
        self.run_graph()

    def print_init_data(self):
        """
        Lists the initial values of the pandemic, as well as
        infection ratio R0.
        :return:
        """
        print("Initial population size: ", self.N)
        print()
        print("Initial susceptible population: ", self.S0)
        print("Initial number of infected: ", self.I0)
        print("Initial number of recovered: ", self.Rec0)
        print("R0: ", self.R0)


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

        print("Calculated number of susceptible individuals "
              "at end of pandemic: ", s_infinite)
        print("Calculated number of recovered individuals "
              "at end of pandemic: ", r_infinite)

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

        plt.plot(self.x_axis, self.susceptible, 'r--',
                 self.x_axis, self.infected, 'b--',
                 self.x_axis, self.recovered, 'g--')

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

        print("Estimated number of susceptible individuals "
              "at end of pandemic: ", self.susceptible[-1])
        print("Estimated number of recovered individuals "
              "at end of pandemic: ", self.recovered[-1])

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

        print ("Highest number of infected: ", max_val)



if __name__ == '__main__':
    model = SIR_Model(6, 5, 500, 5000)

    s_1 = 500 + model.S_prime(500, 1)
    r_1 = 0 + model.R_prime(0, 1)