import numpy as np
from scipy.optimize import minimize
import bayes_logistic as bl
import sys
class Scheme:
    #define parameters
    J = 5 #number of website
    K = 12 #numer of arms (ads)
    d = 12 #dimension of ad features
    simulation_time = 100#length of the simulation
    period_length = 5#length of each period
    Budget = 800000#total budget
    #revised_Budget = 800000 # change the budget because of the budget for each website may not be interger
    mu_true = np.empty([J,K])#true conversion rate
    x = np.array([]) #features of arms

    def __init__(self,period_length = 5):
        self.period_length = period_length

    def generate_data(self, pooled = True, seed = 1):
        '''
        generate x, mu_true
        :param pooled: if generate data in a pooled way
        :param seed: seed of random
        :return: None
        '''
        np.random.seed(seed)
        self.x = np.random.rand(self.K,self.d)*0.1+0.45#random features
        if pooled:#generate data in a pooled way
            beta_bar = np.random.normal(-0.92,0.01,self.d)
            tmp = np.random.rand(self.d,self.d)*0.05
            Sigma = np.dot(tmp,tmp.T)
            for j in range(self.J):
                beta_j = np.random.multivariate_normal(beta_bar,Sigma)
                for k in range(self.K):
                    self.mu_true[j,k] = 1/(1+np.exp(-np.dot(self.x[k],beta_j.T)))
        else:
            self.mu_true = (np.random.rand(self.J,self.K)*5+1.5)/1000
        print "True conversion rate:",self.mu_true
        print "The average conversion rate:",np.average(self.mu_true)

    def simulate(self,seed,scheme = 'TS_HGLM'):
        output = open('result/'+scheme+'_seed_'+str(seed)+'.txt','w')
        if scheme == 'TS_HGLM':#Thompson sampling with a hierarchical generalized linear model (partial pooling)
            np.random.seed(seed)
            prior_beta_bar = np.random.normal(-1,1,self.d)
            tmp = np.random.rand(Scheme.d,Scheme.d)
            prior_Sigma = np.dot(tmp,tmp.T)
            print>>output, prior_beta_bar
            output.flush()


            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])#used to store conversion in each period

            X = np.empty([self.Budget,self.d])#X and y are used to store training data
            y = np.empty([self.Budget])

            for period in range(num_period):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                    for b in range(budget_for_each_website):
                        beta_j = np.random.multivariate_normal(prior_beta_bar,prior_Sigma)#sample a bete_j
                        to_select = 0
                        max = 1/(1+np.exp(-np.dot(self.x[0],beta_j.T)))
                        for i in range(1,self.K):#choose the best arm under sampled beta_j
                            beta_j = np.random.multivariate_normal(prior_beta_bar,prior_Sigma)#sample a bete_j
                            new_value = 1/(1+np.exp(-np.dot(self.x[i],beta_j)))
                            if  new_value > max:
                                to_select, max = i, new_value
                        prob = self.mu_true[j,to_select]
                        X[period*budget_for_each_period + j*budget_for_each_website+b] = self.x[to_select]#all training data is recorded
                        if np.random.random()<prob:
                            y[period*budget_for_each_period + j*budget_for_each_website+b] = 1
                            counter += 1
                        else:
                            y[period*budget_for_each_period + j*budget_for_each_website+b] = 0
                prior_beta_bar, prior_Sigma = bl.fit_bayes_logistic(y[0:(period+1)*budget_for_each_period], X[0:(period+1)*budget_for_each_period], prior_beta_bar, prior_Sigma)
                conversion[period] = counter
                print>>output, prior_beta_bar,counter,np.average(conversion[0:period+1])
                output.flush()
            return np.average(conversion)
        elif scheme == 'Balanced':#balanced allocation
            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            budget_for_each_arm = budget_for_each_website/self.K
            for period in range(num_period):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                    for k in range(self.K):
                        for ii in range(budget_for_each_arm):
                            prob = self.mu_true[j,k]
                            counter += 1 if np.random.random()<prob else 0
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)

        elif scheme == 'Perfect':#scheme with perfect information
            best_arm = np.argmax(self.mu_true,axis = 1)
            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])
            for period in range(num_period):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                        for ii in range(budget_for_each_website):
                            prob = self.mu_true[j,best_arm[j]]
                            counter += 1 if np.random.random()<prob else 0
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Test_rollout_Unpooled':
            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            budget_for_each_arm = budget_for_each_website/self.K

            conversion = np.empty([num_period])
            tau = num_period/5#length of exploration periods

            accumulated_conversion = np.empty([self.J,self.K])
            for period in range(tau):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                    for k in range(self.K):#for each arm
                        for ii in range(budget_for_each_arm):
                            prob = self.mu_true[j,k]
                            if np.random.random()<prob:
                                counter += 1
                                accumulated_conversion[j,k] += 1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])

            best_arm = np.argmax(accumulated_conversion,axis = 1)
            for period in range(tau,num_period):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                        for ii in range(budget_for_each_website):
                            prob = self.mu_true[j,best_arm[j]]
                            counter += 1 if np.random.random()<prob else 0
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Test_rollout_Pooled':
            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            budget_for_each_arm = budget_for_each_website/self.K

            conversion = np.empty([num_period])
            tau = num_period/5#length of exploration

            accumulated_conversion = np.empty([self.K])
            for period in range(tau):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                    for k in range(self.K):#for each arm
                        for ii in range(budget_for_each_arm):
                            prob = self.mu_true[j,k]
                            if np.random.random()<prob:
                                counter += 1
                                accumulated_conversion[k] += 1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])

            best_arm = np.argmax(accumulated_conversion)
            for period in range(tau,num_period):
                counter = 0

                #display ad
                for j in range(self.J):#for each website
                        for ii in range(budget_for_each_website):
                            prob = self.mu_true[j,best_arm]
                            counter += 1 if np.random.random()<prob else 0
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Greedy_Pooled':
            conversion_numerator = np.zeros(self.K)# number of conversions, the same in the following code
            conversion_denominator = np.zeros(self.K)#number of impressions, the same in the following code

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        # 0.00001 is added in denominator to avoid the problem of deviding by zero
                        # 0.00001 is also added in numerator such that the original reward is 1, which guarantees that every arm will be explored
                        # the same in the following code
                        best_arm = np.argmax((conversion_numerator+0.00001)/(conversion_denominator+0.00001))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[best_arm] +=1
                            counter += 1
                        conversion_denominator[best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Greedy_Unpooled':
            conversion_numerator = np.zeros([self.J,self.K])
            conversion_denominator = np.zeros([self.J, self.K])

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator[j]+0.00001)/(conversion_denominator[j]+0.00001))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[j,best_arm] +=1
                            counter += 1
                        conversion_denominator[j,best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Epsilon_Greedy_Pooled_10':
            conversion_numerator = np.zeros(self.K)
            conversion_denominator = np.zeros(self.K)
            conversion_rate = np.zeros(self.K)

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator+0.00001)/(conversion_denominator+0.00001))
                        if np.random.random()<0.1:
                            best_arm = np.random.randint(0,self.K)
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[best_arm] +=1
                            counter += 1
                        conversion_denominator[best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Epsilon_Greedy_Unpooled_10':
            conversion_numerator = np.zeros([self.J,self.K])
            conversion_denominator = np.zeros([self.J, self.K])

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator[j]+0.00001)/(conversion_denominator[j]+0.00001))
                        if np.random.random()<0.1:
                            best_arm = np.random.randint(0,self.K)
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[j,best_arm] +=1
                            counter += 1
                        conversion_denominator[j,best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'UCB1_Pooled':
            conversion_numerator = np.zeros(self.K)#number of conversion
            conversion_denominator = np.zeros(self.K)#number of impression

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(1):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator+0.00001)/(conversion_denominator+0.00001))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[best_arm] +=1
                            counter += 1
                        conversion_denominator[best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            for period in range(1,num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator+0.00001)/(conversion_denominator+0.00001)+np.sqrt(2*np.log(period*budget_for_each_period)/(conversion_numerator+1)))# add 1 to conversion_numerator to avoid the problem of deviding by zeros
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[best_arm] +=1
                            counter += 1
                        conversion_denominator[best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'UCB1_Unpooled':
            conversion_numerator = np.zeros([self.J,self.K])
            conversion_denominator = np.zeros([self.J,self.K])


            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(1):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator[j]+0.00001)/(conversion_denominator[j]+0.00001))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[j,best_arm] +=1
                            counter += 1
                        conversion_denominator[j,best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            for period in range(1,num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator[j]+0.00001)/(conversion_denominator[j]+0.00001)+np.sqrt(2*np.log(period*budget_for_each_website)/(conversion_numerator[j]+1)))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[j,best_arm] +=1
                            counter += 1
                        conversion_denominator[j,best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'UCB1_Tuned_Pooled':
            conversion_numerator = np.zeros(self.K)#number of conversion
            conversion_denominator = np.zeros(self.K)#number of impression

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(1):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator+0.00001)/(conversion_denominator+0.00001))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[best_arm] +=1
                            counter += 1
                        conversion_denominator[best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            for period in range(1,num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        variance = conversion_numerator/conversion_denominator*(1-conversion_numerator/conversion_denominator)
                        V_kt = variance + np.sqrt(2*np.log(period*budget_for_each_period)/(conversion_numerator+1))
                        min = np.minimum(V_kt,np.zeros_like(V_kt)+0.25)
                        best_arm = np.argmax((conversion_numerator+0.00001)/(conversion_denominator+0.00001)+np.sqrt(min*np.log(period*budget_for_each_period)/(conversion_numerator+1)))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[best_arm] +=1
                            counter += 1
                        conversion_denominator[best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'UCB1_Tuned_Unpooled':
            conversion_numerator = np.zeros([self.J,self.K])#number of conversion
            conversion_denominator = np.zeros([self.J,self.K])#number of impression

            num_period = self.simulation_time/self.period_length
            budget_for_each_period = self.Budget/num_period
            budget_for_each_website = budget_for_each_period/self.J
            conversion = np.empty([num_period])

            for period in range(1):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        best_arm = np.argmax((conversion_numerator[j]+0.00001)/(conversion_denominator[j]+0.00001))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[j,best_arm] +=1
                            counter += 1
                        conversion_denominator[j,best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            for period in range(1,num_period):
                counter = 0
                #display ad
                for j in range(self.J):#for each website
                    for ii in range(budget_for_each_website):
                        variance = conversion_numerator[j]/conversion_denominator[j]*(1-conversion_numerator[j]/conversion_denominator[j])
                        V_kt = variance + np.sqrt(2*np.log(period*budget_for_each_website)/(conversion_numerator[j]+1))
                        min = np.minimum(V_kt,np.zeros_like(V_kt)+0.25)
                        best_arm = np.argmax((conversion_numerator[j]+0.00001)/(conversion_denominator[j]+0.00001)+np.sqrt(min*np.log(period*budget_for_each_website)/(conversion_numerator[j]+1)))
                        prob = self.mu_true[j,best_arm]
                        if np.random.random()<prob:
                            conversion_numerator[j,best_arm] +=1
                            counter += 1
                        conversion_denominator[j,best_arm] +=1
                conversion[period] = counter
                print>>output, counter,np.average(conversion[0:period+1])
            return np.average(conversion)
        elif scheme == 'Gittins_Pooled':#TODO
            return 0
            pass
        elif scheme == 'Gittins_Unpooled':#TODO
            return 0
            pass



if __name__ == '__main__':
    scheme = Scheme()
    scheme.generate_data(pooled = True, seed = 1)
    scheme_list = ['Balanced','Perfect','Test_rollout_Unpooled','Test_rollout_Pooled','Greedy_Pooled','Greedy_Unpooled','Epsilon_Greedy_Pooled_10','Epsilon_Greedy_Unpooled_10','UCB1_Pooled','UCB1_Unpooled','UCB1_Tuned_Pooled','UCB1_Tuned_Unpooled','TS_HGLM','Gittins_Pooled','Gittins_Unpooled']
    output = open('combined_result.txt','w')
    for s in scheme_list:
        count = 0
        for seed in range(1,11):
            count += scheme.simulate(seed = seed,scheme = s)
        print>>output,s,count/10
        output.flush()

