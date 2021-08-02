"""
Class ProductComponentModel
Check https://github.com/fernaag/Battery_replacements for latest version.

Methods for handling product-component interactions under different assumptions. 

Created on Mon Jul 05 2021

@authors: Fernando Aguilar Lopez & Romain Billy, NTNU Trondheim, Norway

standard abbreviation: PCM or pcm 

dependencies: #TODO Update this
    numpy >= 1.9
    scipy >= 0.14

Repository for this class, documentation, and tutorials: https://github.com/fernaag/Battery_replacements

"""

import numpy as np
import scipy.stats

def __version__():
    """Return a brief version string and statement for this class."""
    return str('1.0'), str('Class ProductComponentModel, pcm. Version 1.0. Last change: Mon Jul 5th, 2021. Check https://github.com/fernaag/Battery_replacements for latest version.')

class ProductComponentModel(object):
    
    """ Class containing a product component model

    Attributes
    ----------
    t : Series of years or other time intervals

    i_pr : Discrete time series of inflow of product to stock
    i_cm : Discrete time series of inflow of component to stock

    o_pr : Discrete time series of outflow of product from stock
    oc_pr :Discrete time series of outflow of product from stock, by cohort

    o_cm : Discrete time series of outflow of component from stock
    oc_cm :Discrete time series of outflow of component from stock, by cohort

    sc_pr : stock broken down by year and age- cohort for the main product
    sc_cm : stock broken down by year and age- cohort for the component

    s_pr : Discrete time series for product stock, total
    s_cm : Discrete time series for component stock, total

    ds_pr : Discrete time series for product stock change, total
    ds_cm : Discrete time series for component stock change, total

    lt_pr : lifetime distribution of product: dictionary
    lt_cm : lifetime distribution of component: dictionary

    r: Replacement rate of component
    tau_cm: This is the minimum amount of time that a component should remain useful for if it will be eligible for reuse
    tau_pr: This is the minimum amount of time that a product should be expected to remain functional for it to be eligible to get a spare part

    d: Death rate 
    b: Birth rate

    pdf_pr: probability density function of product, distribution of outflow from a specific age-cohort
    pdf_cm: probability density function of component, distribution of outflow from a specific age-cohort
    
    sf_pr: survival function for different age-cohorts of product, year x age-cohort table
    sf_cm: survival function for different age-cohorts of component, year x age-cohort table


    name : string, optional
        Name of the product component model, default is 'PCM'
    """

    """
    Basic initialisation and dimension check methods
    """

    def __init__(self, t=None, i_pr=None, i_cm=None, o_pr=None, o_cm=None, s_pr=None, s_cm=None, lt_pr=None, lt_cm=None, sc_pr=None, sc_cm=None, oc_pr=None, oc_cm=None, name='PCM', pdf_pr=None, pdf_cm=None, sf_pr=None, sf_cm=None, r=None, tau_cm=None, tau_pr=None, d=None, b=None):
        """ Init function. Assign the input data to the instance of the object."""
        self.t = t  # optional

        self.i_pr = i_pr  # optional
        self.i_cm = i_cm # optional

        self.s_pr = s_pr  # optional
        self.sc_pr = sc_pr  # optional

        self.s_cm = s_cm  # optional
        self.sc_cm = sc_cm  # optional

        self.o_pr = o_pr  # optional
        self.oc_pr = oc_pr  # optional

        self.o_cm = o_cm  # optional
        self.oc_cm = oc_cm  # optional

        if lt_pr is not None:
            for ThisKey in lt_pr.keys():
                # If we have the same scalar lifetime, stdDev, etc., for all cohorts,
                # replicate this value to full length of the time vector
                if ThisKey != 'Type':
                    if np.array(lt_pr[ThisKey]).shape[0] == 1:
                        lt_pr[ThisKey] = np.tile(lt_pr[ThisKey], len(t))
        
        if lt_cm is not None:
            for ThisKey in lt_cm.keys():
                # If we have the same scalar lifetime, stdDev, etc., for all cohorts,
                # replicate this value to full length of the time vector
                if ThisKey != 'Type':
                    if np.array(lt_cm[ThisKey]).shape[0] == 1:
                        lt_cm[ThisKey] = np.tile(lt_cm[ThisKey], len(t))

        self.lt_pr = lt_pr  # optional
        self.lt_cm = lt_cm  # optional
        self.name = name  # optional

        self.r = r # optional
        self.tau_cm = tau_cm # optional
        self.tau_pr = tau_pr # optional

        self.d = d # optional
        self.b = b # optional

        self.pdf_pr = pdf_pr # optional
        self.sf_pr  = sf_pr # optional

        self.pdf_cm = pdf_cm # optional
        self.sf_cm  = sf_cm # optional
        

    def compute_stock_change_pr(self):
        """ Determine stock change for product from time series for stock. Formula: stock_change(t) = stock(t) - stock(t-1)."""
        try:
            self.ds_pr = np.zeros(len(self.s_pr))
            self.ds_pr[0] = self.s_pr[0]
            self.ds_pr[1::] = np.diff(self.s_pr)
            return self.ds_pr
        except:
            # Could not determine Stock change of product. The stock is not defined.
            return None     


    def compute_stock_change_cm(self):
        """ Determine stock change for component from time series for stock. Formula: stock_change(t) = stock(t) - stock(t-1)."""
        try:
            self.ds_cm = np.zeros(len(self.s_cm))
            self.ds_cm[0] = self.s_cm[0]
            self.ds_cm[1::] = np.diff(self.s_cm)
            return self.ds_cm
        except:
            # Could not determine Stock change of component. The stock is not defined.
            return None     
        
    def check_stock_balance_pr(self):
        """ Check wether inflow, outflow, and stock are balanced for main product. 
        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change"""
        try:
            Balance = self.i_pr - self.o_pr - self.compute_stock_change_pr()
            return Balance
        except:
            # Could not determine balance. At least one of the variables is not defined.
            return None

    def check_stock_balance_cm(self):
        """ Check wether inflow, outflow, and stock are balanced for the component. 
        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change"""
        try:
            Balance = self.i_cm - self.o_cm - self.compute_stock_change_cm()
            return Balance
        except:
            # Could not determine balance. At least one of the variables is not defined.
            return None
        
    def check_stock_pr_cm(self):
        """ Check if the stock of product and component are the same"""
        try:
            Balance = self.s_pr - self.s_cm 
            return Balance
        except:
            # Could not determine balance. At least one of the variables is not defined.
            return None

    def compute_sf_pr(self): # survival functions
        """
        Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still present at the end of year m (after m-n years).
        The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate scipy function for the lifetime model chosen.
        For lifetimes 0 the sf is also 0, meaning that the age-cohort leaves during the same year of the inflow.
        The method compute outflow_sf returns an array year-by-cohort of the surviving fraction of a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly enters the computation. All other stock variables are determined by mass balance.
        The shape of the output sf array is NoofYears * NoofYears, and the meaning is years by age-cohorts.
        The method does nothing if the sf alreay exists. For example, sf could be assigned to the dynamic stock model from an exogenous computation to save time.
        """
        if self.sf_pr is None:
            self.sf_pr = np.zeros((len(self.t), len(self.t)))
            # Perform specific computations and checks for each lifetime distribution:

            if self.lt_pr['Type'] == 'Fixed': # fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached.
                for m in range(0, len(self.t)):  # cohort index
                    self.sf_pr[m::,m] = np.multiply(1, (np.arange(0,len(self.t)-m) < self.lt_pr['Mean'][m])) # converts bool to 0/1
                # Example: if Lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years, gone after 4 years.

            if self.lt_pr['Type'] == 'Normal': # normally distributed lifetime with mean and standard deviation. Watch out for nonzero values 
                # for negative ages, no correction or truncation done here. Cf. note below.
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_pr[m::,m] = scipy.stats.norm.sf(np.arange(0,len(self.t)-m), loc=self.lt_pr['Mean'][m], scale=self.lt_pr['StdDev'][m])
                        # NOTE: As normal distributions have nonzero pdf for negative ages, which are physically impossible, 
                        # these outflow contributions can either be ignored (violates the mass balance) or
                        # allocated to the zeroth year of residence, the latter being implemented in the method compute compute_o_c_from_s_c.
                        # As alternative, use lognormal or folded normal distribution options.
                        
            if self.lt_pr['Type'] == 'FoldedNormal': # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_pr[m::,m] = scipy.stats.foldnorm.sf(np.arange(0,len(self.t)-m), self.lt_pr['Mean'][m]/self.lt_pr['StdDev'][m], 0, scale=self.lt_pr['StdDev'][m])
                        # NOTE: call this option with the parameters of the normal distribution mu and sigma of curve BEFORE folding,
                        # curve after folding will have different mu and sigma.
                        
            if self.lt_pr['Type'] == 'LogNormal': # lognormal distribution
                # Here, the mean and stddev of the lognormal curve, 
                # not those of the underlying normal distribution, need to be specified! conversion of parameters done here:
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        # calculate parameter mu    of underlying normal distribution:
                        LT_LN = np.log(self.lt_pr['Mean'][m] / np.sqrt(1 + self.lt_pr['Mean'][m] * self.lt_pr['Mean'][m] / (self.lt_pr['StdDev'][m] * self.lt_pr['StdDev'][m]))) 
                        # calculate parameter sigma of underlying normal distribution:
                        SG_LN = np.sqrt(np.log(1 + self.lt_pr['Mean'][m] * self.lt_pr['Mean'][m] / (self.lt_pr['StdDev'][m] * self.lt_pr['StdDev'][m])))
                        # compute survial function
                        self.sf_pr[m::,m] = scipy.stats.lognorm.sf(np.arange(0,len(self.t)-m), s=SG_LN, loc = 0, scale=np.exp(LT_LN)) 
                        # values chosen according to description on
                        # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
                        # Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
                        
            if self.lt_pr['Type'] == 'Weibull': # Weibull distribution with standard definition of scale and shape parameters
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Shape'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_pr[m::,m] = scipy.stats.weibull_min.sf(np.arange(0,len(self.t)-m), c=self.lt_pr['Shape'][m], loc = 0, scale=self.lt_pr['Scale'][m])


            return self.sf_pr
        else:
            # sf already exists
            return self.sf_pr
        
    def compute_sf_cm(self): # survival functions
        """
        Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still present at the end of year m (after m-n years).
        The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate scipy function for the lifetime model chosen.
        For lifetimes 0 the sf is also 0, meaning that the age-cohort leaves during the same year of the inflow.
        The method compute outflow_sf returns an array year-by-cohort of the surviving fraction of a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly enters the computation. All other stock variables are determined by mass balance.
        The shape of the output sf array is NoofYears * NoofYears, and the meaning is years by age-cohorts.
        The method does nothing if the sf alreay exists. For example, sf could be assigned to the dynamic stock model from an exogenous computation to save time.
        """
        if self.sf_cm is None:
            self.sf_cm = np.zeros((len(self.t), len(self.t)))
            # Perform specific computations and checks for each lifetime distribution:

            if self.lt_cm['Type'] == 'Fixed': # fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached.
                for m in range(0, len(self.t)):  # cohort index
                    self.sf_cm[m::,m] = np.multiply(1, (np.arange(0,len(self.t)-m) < self.lt_cm['Mean'][m])) # converts bool to 0/1
                # Example: if Lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years, gone after 4 years.

            if self.lt_cm['Type'] == 'Normal': # normally distributed lifetime with mean and standard deviation. Watch out for nonzero values 
                # for negative ages, no correction or truncation done here. Cf. note below.
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_cm[m::,m] = scipy.stats.norm.sf(np.arange(0,len(self.t)-m), loc=self.lt_cm['Mean'][m], scale=self.lt_cm['StdDev'][m])
                        # NOTE: As normal distributions have nonzero pdf for negative ages, which are physically impossible, 
                        # these outflow contributions can either be ignored (violates the mass balance) or
                        # allocated to the zeroth year of residence, the latter being implemented in the method compute compute_o_c_from_s_c.
                        # As alternative, use lognormal or folded normal distribution options.
                        
            if self.lt_cm['Type'] == 'FoldedNormal': # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_cm[m::,m] = scipy.stats.foldnorm.sf(np.arange(0,len(self.t)-m), self.lt_cm['Mean'][m]/self.lt_cm['StdDev'][m], 0, scale=self.lt_cm['StdDev'][m])
                        # NOTE: call this option with the parameters of the normal distribution mu and sigma of curve BEFORE folding,
                        # curve after folding will have different mu and sigma.
                        
            if self.lt_cm['Type'] == 'LogNormal': # lognormal distribution
                # Here, the mean and stddev of the lognormal curve, 
                # not those of the underlying normal distribution, need to be specified! conversion of parameters done here:
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        # calculate parameter mu    of underlying normal distribution:
                        LT_LN = np.log(self.lt_cm['Mean'][m] / np.sqrt(1 + self.lt_cm['Mean'][m] * self.lt_cm['Mean'][m] / (self.lt_cm['StdDev'][m] * self.lt_cm['StdDev'][m]))) 
                        # calculate parameter sigma of underlying normal distribution:
                        SG_LN = np.sqrt(np.log(1 + self.lt_cm['Mean'][m] * self.lt_cm['Mean'][m] / (self.lt_cm['StdDev'][m] * self.lt_cm['StdDev'][m])))
                        # compute survial function
                        self.sf_cm[m::,m] = scipy.stats.lognorm.sf(np.arange(0,len(self.t)-m), s=SG_LN, loc = 0, scale=np.exp(LT_LN)) 
                        # values chosen according to description on
                        # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
                        # Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
                        
            if self.lt_cm['Type'] == 'Weibull': # Weibull distribution with standard definition of scale and shape parameters
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Shape'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_cm[m::,m] = scipy.stats.weibull_min.sf(np.arange(0,len(self.t)-m), c=self.lt_cm['Shape'][m], loc = 0, scale=self.lt_cm['Scale'][m])


            return self.sf_cm
        else:
            # sf already exists
            return self.sf_cm

    def compute_sf_pr_tau(self): # survival functions for when a delay is needed
        """
        Survival curve for products where t+tau_pr are needed for replacements/reuse
        """
        if self.sf_pr is None:
            self.sf_pr = np.zeros((len(self.t)+self.tau_pr, len(self.t)+self.tau_pr))
            # Perform specific computations and checks for each lifetime distribution:

            if self.lt_pr['Type'] == 'Fixed': # fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached.
                tau_array =  np.ones(self.tau_pr) * self.lt_pr['Mean'][-1] # Create an array with the last mean value for the length of tau_pr
                self.lt_pr['Mean'] = np.concatenate((self.lt_pr['Mean'],tau_array))
                for m in range(0, len(self.t)+self.tau_pr):  # cohort index
                    self.sf_pr[m::,m] = np.multiply(1, (np.arange(0,len(self.t)+self.tau_pr-m) < self.lt_pr['Mean'][m])) # converts bool to 0/1
                # Example: if Lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years, gone after 4 years.

            if self.lt_pr['Type'] == 'Normal': # normally distributed lifetime with mean and standard deviation. Watch out for nonzero values 
                # for negative ages, no correction or truncation done here. Cf. note below.
                tau_mean_array =  np.ones(self.tau_pr) * self.lt_pr['Mean'][-1] # Create an array with the last mean value for the length of tau_pr
                tau_stdev_array =  np.ones(self.tau_pr) * self.lt_pr['StdDev'][-1] # Create an array with the last mean value for the length of tau_pr
                self.lt_pr['Mean'] = np.concatenate((self.lt_pr['Mean'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_pr
                self.lt_pr['StdDev'] = np.concatenate((self.lt_pr['StdDev'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_pr
                for m in range(0, len(self.t)+self.tau_pr):  # cohort index
                    if self.lt_pr['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_pr[m::,m] = scipy.stats.norm.sf(np.arange(0,len(self.t)+self.tau_pr-m), loc=self.lt_pr['Mean'][m], scale=self.lt_pr['StdDev'][m])
                        # NOTE: As normal distributions have nonzero pdf for negative ages, which are physically impossible, 
                        # these outflow contributions can either be ignored (violates the mass balance) or
                        # allocated to the zeroth year of residence, the latter being implemented in the method compute compute_o_c_from_s_c.
                        # As alternative, use lognormal or folded normal distribution options. 
                        
            if self.lt_pr['Type'] == 'FoldedNormal': # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
                tau_mean_array =  np.ones(self.tau_pr) * self.lt_pr['Mean'][-1] # Create an array with the last mean value for the length of tau_pr
                tau_stdev_array =  np.ones(self.tau_pr) * self.lt_pr['StdDev'][-1] # Create an array with the last mean value for the length of tau_pr
                self.lt_pr['Mean'] = np.concatenate((self.lt_pr['Mean'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_pr
                self.lt_pr['StdDev'] = np.concatenate((self.lt_pr['StdDev'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_pr
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_pr[m::,m] = scipy.stats.foldnorm.sf(np.arange(0,len(self.t)-m), self.lt_pr['Mean'][m]/self.lt_pr['StdDev'][m], 0, scale=self.lt_pr['StdDev'][m])
                        # NOTE: call this option with the parameters of the normal distribution mu and sigma of curve BEFORE folding,
                        # curve after folding will have different mu and sigma.
                        
            if self.lt_pr['Type'] == 'LogNormal': # lognormal distribution
                tau_mean_array =  np.ones(self.tau_pr) * self.lt_pr['Mean'][-1] # Create an array with the last mean value for the length of tau_pr
                tau_stdev_array =  np.ones(self.tau_pr) * self.lt_pr['StdDev'][-1] # Create an array with the last mean value for the length of tau_pr
                self.lt_pr['Mean'] = np.concatenate((self.lt_pr['Mean'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_pr
                self.lt_pr['StdDev'] = np.concatenate((self.lt_pr['StdDev'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_pr
                # Here, the mean and stddev of the lognormal curve, 
                # not those of the underlying normal distribution, need to be specified! conversion of parameters done here:
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        # calculate parameter mu    of underlying normal distribution:
                        LT_LN = np.log(self.lt_pr['Mean'][m] / np.sqrt(1 + self.lt_pr['Mean'][m] * self.lt_pr['Mean'][m] / (self.lt_pr['StdDev'][m] * self.lt_pr['StdDev'][m]))) 
                        # calculate parameter sigma of underlying normal distribution:
                        SG_LN = np.sqrt(np.log(1 + self.lt_pr['Mean'][m] * self.lt_pr['Mean'][m] / (self.lt_pr['StdDev'][m] * self.lt_pr['StdDev'][m])))
                        # compute survial function
                        self.sf_pr[m::,m] = scipy.stats.lognorm.sf(np.arange(0,len(self.t)-m), s=SG_LN, loc = 0, scale=np.exp(LT_LN)) 
                        # values chosen according to description on
                        # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
                        # Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
                        
            if self.lt_pr['Type'] == 'Weibull': # Weibull distribution with standard definition of scale and shape parameters
                tau_mean_array =  np.ones(self.tau_pr) * self.lt_pr['Shape'][-1] # Create an array with the last mean value for the length of tau_pr
                tau_stdev_array =  np.ones(self.tau_pr) * self.lt_pr['Scale'][-1] # Create an array with the last mean value for the length of tau_pr
                self.lt_pr['Shape'] = np.concatenate((self.lt_pr['Shape'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_pr
                self.lt_pr['Scale'] = np.concatenate((self.lt_pr['Scale'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_pr
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_pr['Shape'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_pr[m::,m] = scipy.stats.weibull_min.sf(np.arange(0,len(self.t)-m), c=self.lt_pr['Shape'][m], loc = 0, scale=self.lt_pr['Scale'][m])


            return self.sf_pr
        else:
            # sf already exists
            return self.sf_pr


    def compute_sf_cm_tau(self): # survival functions for when a delay is needed
        """
        Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still present at the end of year m (after m-n years).
        The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate scipy function for the lifetime model chosen.
        For lifetimes 0 the sf is also 0, meaning that the age-cohort leaves during the same year of the inflow.
        The method compute outflow_sf returns an array year-by-cohort of the surviving fraction of a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly enters the computation. All other stock variables are determined by mass balance.
        The shape of the output sf array is NoofYears * NoofYears, and the meaning is years by age-cohorts.
        The method does nothing if the sf alreay exists. For example, sf could be assigned to the dynamic stock model from an exogenous computation to save time.
        """
        
        if self.sf_cm is None:
            self.sf_cm = np.zeros((len(self.t)+self.tau_cm, len(self.t)+self.tau_cm))
            # Perform specific computations and checks for each lifetime distribution:

            if self.lt_cm['Type'] == 'Fixed': # fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached.
                tau_array =  np.ones(self.tau_cm) * self.lt_cm['Mean'][-1] # Create an array with the last mean value for the length of tau_cm
                self.lt_cm['Mean'] = np.concatenate((self.lt_cm['Mean'],tau_array))
                for m in range(0, len(self.t)+self.tau_cm):  # cohort index
                    self.sf_cm[m::,m] = np.multiply(1, (np.arange(0,len(self.t)+self.tau_cm-m) < self.lt_cm['Mean'][m])) # converts bool to 0/1
                # Example: if Lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years, gone after 4 years.

            if self.lt_cm['Type'] == 'Normal': # normally distributed lifetime with mean and standard deviation. Watch out for nonzero values 
                # for negative ages, no correction or truncation done here. Cf. note below.
                tau_mean_array =  np.ones(self.tau_cm) * self.lt_cm['Mean'][-1] # Create an array with the last mean value for the length of tau_cm
                tau_stdev_array =  np.ones(self.tau_cm) * self.lt_cm['StdDev'][-1] # Create an array with the last mean value for the length of tau_cm
                self.lt_cm['Mean'] = np.concatenate((self.lt_cm['Mean'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_cm
                self.lt_cm['StdDev'] = np.concatenate((self.lt_cm['StdDev'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_cm
                for m in range(0, len(self.t)+self.tau_cm):  # cohort index
                    if self.lt_cm['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_cm[m::,m] = scipy.stats.norm.sf(np.arange(0,len(self.t)+self.tau_cm-m), loc=self.lt_cm['Mean'][m], scale=self.lt_cm['StdDev'][m])
                        # NOTE: As normal distributions have nonzero pdf for negative ages, which are physically impossible, 
                        # these outflow contributions can either be ignored (violates the mass balance) or
                        # allocated to the zeroth year of residence, the latter being implemented in the method compute compute_o_c_from_s_c.
                        # As alternative, use lognormal or folded normal distribution options.
                        
            if self.lt_cm['Type'] == 'FoldedNormal': # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
                tau_mean_array =  np.ones(self.tau_cm) * self.lt_cm['Mean'][-1] # Create an array with the last mean value for the length of tau_cm
                tau_stdev_array =  np.ones(self.tau_cm) * self.lt_cm['StdDev'][-1] # Create an array with the last mean value for the length of tau_cm
                self.lt_cm['Mean'] = np.concatenate((self.lt_cm['Mean'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_cm
                self.lt_cm['StdDev'] = np.concatenate((self.lt_cm['StdDev'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_cm
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_cm[m::,m] = scipy.stats.foldnorm.sf(np.arange(0,len(self.t)-m), self.lt_cm['Mean'][m]/self.lt_cm['StdDev'][m], 0, scale=self.lt_cm['StdDev'][m])
                        # NOTE: call this option with the parameters of the normal distribution mu and sigma of curve BEFORE folding,
                        # curve after folding will have different mu and sigma.
                        
            if self.lt_cm['Type'] == 'LogNormal': # lognormal distribution
                tau_mean_array =  np.ones(self.tau_cm) * self.lt_cm['Mean'][-1] # Create an array with the last mean value for the length of tau_cm
                tau_stdev_array =  np.ones(self.tau_cm) * self.lt_cm['StdDev'][-1] # Create an array with the last mean value for the length of tau_cm
                self.lt_cm['Mean'] = np.concatenate((self.lt_cm['Mean'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_cm
                self.lt_cm['StdDev'] = np.concatenate((self.lt_cm['StdDev'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_cm
                # Here, the mean and stddev of the lognormal curve, 
                # not those of the underlying normal distribution, need to be specified! conversion of parameters done here:
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Mean'][m] != 0:  # For products with lifetime of 0, sf == 0
                        # calculate parameter mu    of underlying normal distribution:
                        LT_LN = np.log(self.lt_cm['Mean'][m] / np.sqrt(1 + self.lt_cm['Mean'][m] * self.lt_cm['Mean'][m] / (self.lt_cm['StdDev'][m] * self.lt_cm['StdDev'][m]))) 
                        # calculate parameter sigma of underlying normal distribution:
                        SG_LN = np.sqrt(np.log(1 + self.lt_cm['Mean'][m] * self.lt_cm['Mean'][m] / (self.lt_cm['StdDev'][m] * self.lt_cm['StdDev'][m])))
                        # compute survial function
                        self.sf_cm[m::,m] = scipy.stats.lognorm.sf(np.arange(0,len(self.t)-m), s=SG_LN, loc = 0, scale=np.exp(LT_LN)) 
                        # values chosen according to description on
                        # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
                        # Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
                        
            if self.lt_cm['Type'] == 'Weibull': # Weibull distribution with standard definition of scale and shape parameters
                tau_mean_array =  np.ones(self.tau_cm) * self.lt_cm['Shape'][-1] # Create an array with the last mean value for the length of tau_cm
                tau_stdev_array =  np.ones(self.tau_cm) * self.lt_cm['Scale'][-1] # Create an array with the last mean value for the length of tau_cm
                self.lt_cm['Shape'] = np.concatenate((self.lt_cm['Shape'],tau_mean_array)) # Expand the size of the Mean array with last value by tau_cm
                self.lt_cm['Scale'] = np.concatenate((self.lt_cm['Scale'],tau_stdev_array)) # Expand the size of the StDev array with last value by tau_cm
                for m in range(0, len(self.t)):  # cohort index
                    if self.lt_cm['Shape'][m] != 0:  # For products with lifetime of 0, sf == 0
                        self.sf_cm[m::,m] = scipy.stats.weibull_min.sf(np.arange(0,len(self.t)-m), c=self.lt_cm['Shape'][m], loc = 0, scale=self.lt_cm['Scale'][m])


            return self.sf_cm
        else:
            # sf already exists
            return self.sf_cm

    def case_1(self):
        '''
        Products have a lifetime that includes all types of failures, component EoL being one of them. No component replacement. 1 product = one component, Outflow component = outflow product 

        Since this case includes all possible failures, the choice of lifetime should not be limted to the technical lifetime of the product itself but should also consider how this might be 
        affected by failures in the component and potential damages. Therefore, the probability that a failure occurs is greater than the techical lifetime of the product itself and should lead
        to a shorter choice of lifetime. 

        This is a simpler and less explicit method for accounting for failures of both products with a single lifetime. 
        '''
        if self.s_pr is not None:
            if self.lt_pr is not None: 
                self.sc_pr = np.zeros((len(self.t), len(self.t)))
                self.oc_pr = np.zeros((len(self.t), len(self.t)))
                self.i_pr = np.zeros(len(self.t))
                self.sc_cm = np.zeros((len(self.t), len(self.t)))
                self.oc_cm = np.zeros((len(self.t), len(self.t)))
                self.i_cm = np.zeros(len(self.t))
                # construct the sf of a product of cohort tc remaining in the stock in year t
                self.compute_sf_pr() # Computes sf if not present already.
                if self.sf_pr[0, 0] != 0: # Else, inflow is 0.
                    self.i_pr[0] = self.s_pr[0] / self.sf_pr[0, 0]
                self.sc_pr[:, 0] = self.i_pr[0] * self.sf_pr[:, 0] # Future decay of age-cohort of year 0.
                self.oc_pr[0, 0] = self.i_pr[0] - self.sc_pr[0, 0]
                # all other years:
                for m in range(1, len(self.t)):  # for all years m, starting in second year
                    # 1) Compute outflow from previous age-cohorts up to m-1
                    self.oc_pr[m, 0:m] = self.sc_pr[m-1, 0:m] - self.sc_pr[m, 0:m] # outflow table is filled row-wise, for each year m.
                    # 2) Determine inflow from mass balance:
                    
                    if self.sf_pr[m,m] != 0: # Else, inflow is 0.
                        self.i_pr[m] = (self.s_pr[m] - self.sc_pr[m, :].sum()) / self.sf_pr[m,m] # allow for outflow during first year by rescaling with 1/sf[m,m]
                    # 3) Add new inflow to stock and determine future decay of new age-cohort
                    self.sc_pr[m::, m] = self.i_pr[m] * self.sf_pr[m::, m]
                    self.oc_pr[m, m]   = self.i_pr[m] * (1 - self.sf_pr[m, m])
                # 4) Determining the values for the component
                self.sc_cm = self.sc_pr
                self.oc_cm = self.oc_pr 
                self.s_cm = self.s_pr
                self.i_cm = self.i_pr
                # Calculating total values
                self.o_pr = self.oc_pr.sum(axis=1)
                self.o_cm = self.oc_cm.sum(axis=1)
                    
                return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.oc_pr, self.oc_cm
            else:
                raise Exception('No lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None
                
    def case_2(self):
        '''
        Products have a lifetime that includes all types of failures, component EoL being one of them. Still, it is assumed that some components will be replaced at a given rate r. 
        More than one component is used in the lifetime of the product. Outflow component >= outflow product

        In this case the failure of components is accounted for by a replacement rate and therefore the lifetime of the vehicle can be assumed to not be affected to the same extent as in case 1. 
        Limitations to this approach might be that the outflows and component inflows are estimated at different times as the case where a separate lifetime would be calculated for the components. 
        It is more suitable if the stock is approximately constant.  
        '''
        if self.s_pr is not None:
            if self.lt_pr is not None:
                if self.r is not None: 
                    self.sc_pr = np.zeros((len(self.t), len(self.t)))
                    self.oc_pr = np.zeros((len(self.t), len(self.t)))
                    self.i_pr = np.zeros(len(self.t))

                    self.i_cm = np.zeros(len(self.t))
                    # construct the sf of a product of cohort tc remaining in the stock in year t
                    self.compute_sf_pr() # Computes sf if not present already.
                    if self.sf_pr[0, 0] != 0: # Else, inflow is 0.
                        self.i_pr[0] = self.s_pr[0] / self.sf_pr[0, 0]
                    self.sc_pr[:, 0] = self.i_pr[0] * self.sf_pr[:, 0] # Future decay of age-cohort of year 0.
                    self.oc_pr[0, 0] = self.i_pr[0] - self.sc_pr[0, 0]
                    # all other years:
                    for m in range(1, len(self.t)):  # for all years m, starting in second year
                        # 1) Compute outflow from previous age-cohorts up to m-1
                        self.oc_pr[m, 0:m] = self.sc_pr[m-1, 0:m] - self.sc_pr[m, 0:m] # outflow table is filled row-wise, for each year m.
                        # 2) Determine inflow from mass balance:
                        
                        if self.sf_pr[m,m] != 0: # Else, inflow is 0.
                            self.i_pr[m] = (self.s_pr[m] - self.sc_pr[m, :].sum()) / self.sf_pr[m,m] # allow for outflow during first year by rescaling with 1/sf[m,m]
                        # 3) Add new inflow to stock and determine future decay of new age-cohort
                        self.sc_pr[m::, m] = self.i_pr[m] * self.sf_pr[m::, m]
                        self.oc_pr[m, m]   = self.i_pr[m] * (1 - self.sf_pr[m, m])
                    # 4) Determining the values for the component
                    self.s_cm = self.s_pr
                    self.i_cm = self.i_pr * (1 + self.r)
                    self.o_cm = self.i_cm - self.compute_stock_change_cm()

                    # Calculating total values
                    self.o_pr = self.oc_pr.sum(axis=1)

                    return self.sc_pr, self.i_pr, self.i_cm, self.oc_pr
                else:
                    raise Exception('No replacement rate specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None

    def case_3(self):
        '''
        Products and components have independent lifetimes. Potential failure of the component is not included in the lifetime of the product. Components can neither be replaced nor reused, meaning that if 
        either the product or the component fail, they are both scrapped.

        In this case, the combined lifetimes of the component and the product yield a smaller effective lifetime that dictates the outflows based on all failures. 
        This would be a similar case as 1, but the effective lifetime choice is more explicit and potentially more robust, since it considers aspects of both products. 

        To avoid double-counting of outflows in the case that a product and component falure occur at the same time, we adjust the real stock with a virtual one after the outflows of the product have occurred.
        This means that when there is a risk of double counting, the outflows are attributed to product failures. 

        The real outflows in this case are equal to the product attributed outflows + the component attributed outflows.
        '''
        if self.s_pr is not None:
            if self.lt_pr is not None: 
                if self.lt_cm is not None:
                    self.sc_pr = np.zeros((len(self.t), len(self.t)))
                    self.oc_pr = np.zeros((len(self.t), len(self.t)))
                    self.i_pr = np.zeros(len(self.t))
                    self.sc_cm = np.zeros((len(self.t), len(self.t)))
                    self.oc_cm = np.zeros((len(self.t), len(self.t)))
                    self.i_cm = np.zeros(len(self.t))
                    self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                    self.o_cm = np.zeros(len(self.t))
                    self.o_pr = np.zeros(len(self.t))
                    # Initializing values
                    self.sc_pr[0,0] = self.s_pr[0]
                    self.o_pr[0] = 0 
                    self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
                    self.o_pr[1] = 0
                    self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
                    self.sc_pr[1,1] = self.i_pr[1]

                    # construct the sf of a product of cohort tc remaining in the stock in year t
                    self.compute_sf_pr() # Computes sf if not present already.
                    self.compute_sf_cm() # Computes sf of component if not present already.
                    # all other years:            
                    for m in range(1, len(self.t)):  # for all years m, starting in second year
                        # 1) Compute outflow from previous age-cohorts up to m-1
                        if self.sf_pr[m,m] != 0 and self.sf_cm[m,m] != 0: # Else, inflow is 0.
                            self.oc_pr[m, 0:m] = self.sc_pr[m-1, 0:m]/self.sf_pr[m-1,0:m] * abs((self.sf_pr[m, 0:m] - self.sf_pr[m-1, 0:m]))  # Calculating outflows attributed to product failures
                            self.oc_cm[m, 0:m] = (self.sc_pr[m-1, 0:m] - self.oc_pr[m, 0:m])/self.sf_cm[m-1,0:m] * abs((self.sf_cm[m, 0:m] - self.sf_cm[m-1, 0:m]))# Calculating outflows attributed to component failures
                            self.sc_pr[m,0:m] = self.sc_pr[m-1,:m] - self.oc_pr[m, 0:m] - self.oc_cm[m, 0:m] # Computing real stock
                            self.oc_pr[m, 0:m] = self.oc_pr[m, 0:m] + self.oc_cm[m, 0:m]
                            self.oc_cm[m, 0:m] = self.oc_pr[m, 0:m]
                            # 2) Determine inflow from mass balance:
                             #/ self.sf_pr[m,m] # allow for outflow during first year by rescaling with 1/sf[m,m]
                            #self.i_cm[m] = self.ds_pr[m] +  self.oc_cm.sum(axis=1)[m]
                        # 3) Add new inflow to stock and determine future decay of new age-cohort
                        self.i_pr[m] = self.ds_pr[m] + self.oc_pr.sum(axis=1)[m] 
                        self.i_cm[m] = self.i_pr[m]
                        self.sc_pr[m,m] = self.i_pr[m]
                    self.o_pr = self.oc_pr.sum(axis=1)
                    self.o_cm = self.oc_cm.sum(axis=1)
                    self.sc_cm = self.sc_pr
                    self.s_cm = self.s_pr
                    # return self.sc_pr, self.sc_cm 
                else:
                    raise Exception('No component lifetime specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No product lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None
                
    def case_4(self):
        '''
        ## Case 4 
        Products and components have independent lifetimes. Potential failure of the component is not included in the lifetime of the product. 
        Components cannot be replaced, but they can be reused. If the component fails, the product is scrapped. 
        If the product fails but the status of the component is still good, it can be reused in a new product. 
        This case does not necessarily make sense for cars, but could be useful for other applications.

        To determine the amount of components that have gone out, we take the outflows due to vehicle failures and use the probability of the component still being useful 
        from the component survival function with a delay tau to ensure that we don't reuse components that will fail soon anyway. Since we attribute simultanious failures 
        to the components anyway, we should be onm the conservative side of the potential components for reuse. 
        '''
        if self.s_pr is not None:
            if self.lt_pr is not None: 
                if self.lt_cm is not None:
                    if self.tau_cm is not None:
                        self.sc_pr = np.zeros((len(self.t), len(self.t)))
                        self.oc_pr = np.zeros((len(self.t), len(self.t)))
                        self.i_pr = np.zeros(len(self.t))
                        self.sc_cm = np.zeros((len(self.t), len(self.t)))
                        self.oc_cm = np.zeros((len(self.t), len(self.t)))
                        self.i_cm = np.zeros(len(self.t))
                        self.o_cm = np.zeros(len(self.t))
                        self.o_pr = np.zeros(len(self.t))
                        self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                        reuse = np.zeros((len(self.t), len(self.t)))

                        # Initializing values
                        self.sc_pr[0,0] = self.s_pr[0]
                        self.o_pr[0] = 0 
                        self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
                        self.o_pr[1] = 0
                        self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
                        self.sc_pr[1,1] = self.i_pr[1]
                        
                        self.sc_cm[0,0] = self.s_pr[0]
                        self.o_cm[0] = 0 
                        self.i_cm[0] = self.ds_pr[0] - self.o_cm[0]
                        self.o_cm[1] = 0
                        self.i_cm[1] = self.ds_pr[1] - self.o_cm[1]
                        self.sc_cm[1,1] = self.i_cm[1]

                        # construct the sf of a product of cohort tc remaining in the stock in year t
                        self.compute_sf_pr() # Computes sf if not present already.
                        self.compute_sf_cm_tau() # Computes sf od component if not present already.

                        for m in range(1, len(self.t)):  # for all years m, starting in second year
                            # 1) Compute outflow from previous age-cohorts up to m-1
                            if self.sf_pr[m,m] != 0 and self.sf_cm[m,m] != 0: # Else, inflow is 0.
                                self.oc_cm[m, 0:m] = self.sc_cm[m-1, 0:m]/self.sf_cm[m-1,0:m] * abs((self.sf_cm[m, 0:m] - self.sf_cm[m-1, 0:m]))# FIXME: Truedev gives a warning for some values, TODO: Double-check if use of sc_cm is correct here rather than sc_pr
                                self.oc_pr[m, 0:m] = (self.sc_pr[m-1, 0:m] - self.oc_cm[m, 0:m])/self.sf_pr[m-1,0:m] * abs((self.sf_pr[m, 0:m] - self.sf_pr[m-1, 0:m]))  # Calculating outflows attributed to product failures
                                # defining the share of components that is useful TODO: Is this really sf_cm we need to use?
                                reuse[m,:m] = self.oc_pr[m,0:m] * self.sf_cm[m+self.tau_cm, 0:m]
                                self.oc_pr[m,0:m] = self.oc_pr[m, 0:m] + self.oc_cm[m, 0:m]
                                self.oc_cm[m,0:m] = self.oc_pr[m, 0:m] - reuse[m,:m]
                                self.sc_pr[m,0:m] = self.sc_pr[m-1,0:m] - self.oc_pr[m, 0:m]  # Computing real stock
                                self.sc_cm[m,0:m] = self.sc_cm[m-1,0:m] - self.oc_cm[m,0:m] 
                            self.i_pr[m] = self.ds_pr[m] + self.oc_pr.sum(axis=1)[m] 
                            self.i_cm[m] = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
                            self.sc_pr[m,m] = self.i_pr[m]
                            self.sc_cm[m,m] = self.i_cm[m]
                        self.o_pr = self.oc_pr.sum(axis=1)
                        self.o_cm = self.oc_cm.sum(axis=1)
                        self.s_pr = self.sc_pr.sum(axis=1)
                        self.s_cm = self.sc_cm.sum(axis=1)
                        #return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.oc_pr, self.oc_cm
                    else:
                        raise Exception('No delay specified')
                        return None, None, None, None, None, None
                else:
                    raise Exception('No component lifetime specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No product lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None
            

    def case_5(self):
        '''
        Products and components have independent lifetimes. Potential failure of the component is not included in the lifetime of the product. 
        Components can be replaced, but they cannot be reused. If the product fails, the component is scrapped. If the component fails but the product is still in good shape, 
        a new components can be used to make the product operational again. Since the lifetimes of the product and the components are independent, replacing the component should have no effect on 
        the product’s lifetime. 

        In this case the lifetime of the product is not limited by the lifetime of a component. 
        Without an additional logic, this means that component replacements will always happen to allow the vehicle to live the full extent of it's expected lifetime. 
        As a consequence, there can be components that are barely used in the fleet that are being scrapped wiht the vehicle. 
        To avoid this, we could again rely on the survival function to only replace components that are either under warranty or that will stay in the fleet for a reasonable amount of time. 

        We need to replace the components that are failing and put new ones in products that are still in the stock. 
        Since there are some products that fail simultainiously as the components, we should not replace all of the component attributed outflows. 
        We therefore attribute the simultanious failures to products instead of components and use the sf of the components as a share of components that would still be in good condition for reused

        The components are put in products of the same cohort or older. Newer cohorts cannot get older components. 

        Double counting is attributed to the product to make sure that only functional products are considered for replacement
        '''
        if self.s_pr is not None:
            if self.lt_pr is not None: 
                if self.lt_cm is not None:
                    if self.tau_pr is not None:
                        self.sc_pr = np.zeros((len(self.t), len(self.t)))
                        self.oc_pr = np.zeros((len(self.t), len(self.t)))
                        self.i_pr = np.zeros(len(self.t))
                        self.sc_cm = np.zeros((len(self.t), len(self.t)))
                        self.oc_cm = np.zeros((len(self.t), len(self.t)))
                        self.i_cm = np.zeros(len(self.t))
                        self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                        self.o_cm = np.zeros(len(self.t))
                        self.o_pr = np.zeros(len(self.t))
                        replacement = np.zeros((len(self.t), len(self.t)))
                        # Initializing values
                        self.sc_pr[0,0] = self.s_pr[0]
                        self.o_pr[0] = 0 
                        self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
                        self.o_pr[1] = 0
                        self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
                        self.sc_pr[1,1] = self.i_pr[1]
                        
                        self.sc_cm[0,0] = self.s_pr[0]
                        self.o_cm[0] = 0 
                        self.i_cm[0] = self.ds_pr[0] - self.o_pr[0]
                        self.o_cm[1] = 0
                        self.i_cm[1] = self.ds_pr[1] - self.o_pr[1]
                        self.sc_cm[1,1] = self.i_pr[1]

                        self.compute_sf_pr_tau() # Computes sf if not present already.
                        # Since now we need the share of components that will still be useful in tau_cm years, 
                        # we need to compute the sf_cm curve for the length of t+tau_cm
                        self.compute_sf_cm() # Computes sf od component if not present already.
                        # all other years:            
                        for m in range(1, len(self.t)):  # for all years m, starting in second year
                            for c in range(m+1):
                            # 1) Compute outflow from previous age-cohorts up to m-1
                                if  self.sf_cm[m-1,c] != 0 and self.sf_pr[m-1,c] != 0: # Else, inflow is 0.
                                    self.oc_pr[m, c] = self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c]))  # alculating outflows attributed to product failures
                                    # FIXME: I take here sc_cm instead of sc_pr since I think that the outflows of batteries should follow the cohort composition of the stock of batteries, not of cars. @Romain: Does this make sense? 
                                    self.oc_cm[m, c] = (self.sc_cm[m-1, c]- self.oc_pr[m, c])/self.sf_cm[m-1,c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))# Calculating outflows attributed to component failures 
                                    # Defining the amount of products eligible for component replacement 
                                    replacement[m,c] = (self.sf_pr[m+self.tau_pr, c]) * self.oc_cm[m, c]
                                    # Correcting outflows0
                                    self.oc_pr[m, c] = self.oc_pr[m, c]+ self.oc_cm[m, c]  - replacement[m,c]
                                    self.oc_cm[m, c] = self.oc_cm[m, c] +  self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c]))
                                    self.sc_pr[m,c] = self.sc_pr[m-1,c] - self.oc_pr[m, c]  # Computing real stock
                                    self.sc_cm[m,c] = self.sc_cm[m-1,c] - self.oc_cm[m,c]
                                self.i_pr[m] = self.ds_pr[m] + self.oc_pr.sum(axis=1)[m] 
                                self.i_cm[m] = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
                                self.sc_pr[m,m] = self.i_pr[m]
                                self.sc_cm[m,m] = self.i_cm[m]
                        #return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.oc_pr, self.oc_cm
                    else:
                        raise Exception('No delay specified')
                        return None, None, None, None, None, None
                else:
                    raise Exception('No component lifetime specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No product lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None

    # This code was used before and is no longer needed, if we find that the new version above is correct.
    # def case_5(self):
    #     '''
    #     Products and components have independent lifetimes. Potential failure of the component is not included in the lifetime of the product. 
    #     Components can be replaced, but they cannot be reused. If the product fails, the component is scrapped. If the component fails but the product is still in good shape, 
    #     a new components can be used to make the product operational again. Since the lifetimes of the product and the components are independent, replacing the component should have no effect on 
    #     the product’s lifetime. 

    #     In this case the lifetime of the product is not limited by the lifetime of a component. 
    #     Without an additional logic, this means that component replacements will always happen to allow the vehicle to live the full extent of it's expected lifetime. 
    #     As a consequence, there can be components that are barely used in the fleet that are being scrapped wiht the vehicle. 
    #     To avoid this, we could again rely on the survival function to only replace components that are either under warranty or that will stay in the fleet for a reasonable amount of time. 

    #     We need to replace the components that are failing and put new ones in products that are still in the stock. 
    #     Since there are some products that fail simultainiously as the components, we should not replace all of the component attributed outflows. 
    #     We therefore attribute the simultanious failures to products instead of components and use the sf of the components as a share of components that would still be in good condition for reused

    #     The components are put in products of the same cohort or older. Newer cohorts cannot get older components. 

    #     Double counting is attributed to the product to make sure that only functional products are considered for replacement
    #     '''
    #     if self.s_pr is not None:
    #         if self.lt_pr is not None: 
    #             if self.lt_cm is not None:
    #                 if self.tau_pr is not None:
    #                     self.sc_pr = np.zeros((len(self.t), len(self.t)))
    #                     self.oc_pr = np.zeros((len(self.t), len(self.t)))
    #                     self.i_pr = np.zeros(len(self.t))
    #                     self.sc_cm = np.zeros((len(self.t), len(self.t)))
    #                     self.oc_cm = np.zeros((len(self.t), len(self.t)))
    #                     self.i_cm = np.zeros(len(self.t))
    #                     self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
    #                     self.o_cm = np.zeros(len(self.t))
    #                     self.o_pr = np.zeros(len(self.t))
    #                     replacement = np.zeros((len(self.t), len(self.t)))
    #                     # Initializing values
    #                     self.sc_pr[0,0] = self.s_pr[0]
    #                     self.o_pr[0] = 0 
    #                     self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
    #                     self.o_pr[1] = 0
    #                     self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
    #                     self.sc_pr[1,1] = self.i_pr[1]

    #                     self.compute_sf_pr_tau() # Computes sf if not present already.
    #                     # Since now we need the share of components that will still be useful in tau_cm years, 
    #                     # we need to compute the sf_cm curve for the length of t+tau_cm
    #                     self.compute_sf_cm() # Computes sf od component if not present already.
    #                     # all other years:            
    #                     for m in range(1, len(self.t)):  # for all years m, starting in second year
    #                         for c in range(m+1):
    #                         # 1) Compute outflow from previous age-cohorts up to m-1
    #                             if  self.sf_cm[m-1,c] != 0 and self.sf_pr[m-1,c] != 0: # Else, inflow is 0.
    #                                 self.oc_pr[m, c] = self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c]))  # alculating outflows attributed to product failures
    #                                 self.oc_cm[m, c] = (self.sc_pr[m-1, c]- self.oc_pr[m, c])/self.sf_cm[m-1,c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))# Calculating outflows attributed to component failures 
    #                                 # Defining the amount of products eligible for component replacement 
    #                                 replacement[m,c] = (self.sf_pr[m+self.tau_pr, c]) * self.oc_cm[m, c]
    #                                 # Correcting outflows0
    #                                 self.oc_pr[m, c] = self.oc_pr[m, c]+ self.oc_cm[m, c]  - replacement[m,c]
    #                                 self.oc_cm[m, c] = self.oc_pr[m, c] +  replacement[m,c]
    #                                 self.sc_pr[m,c] = self.sc_pr[m-1,c] - self.oc_pr[m, c]  # Computing real stock
    #                                 self.sc_cm[m,c] = self.sc_cm[m-1,c] - self.oc_cm[m,c]
    #                             self.i_pr[m] = self.ds_pr[m] + self.oc_pr.sum(axis=1)[m] 
    #                             self.i_cm[m] = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
    #                             self.sc_pr[m,m] = self.i_pr[m]
    #                         # TODO: Need to add stock by cohort of batteries

    #                     return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.oc_pr, self.oc_cm
    #                 else:
    #                     raise Exception('No delay specified')
    #                     return None, None, None, None, None, None
    #             else:
    #                 raise Exception('No component lifetime specified')
    #                 return None, None, None, None, None, None
    #         else:
    #             raise Exception('No product lifetime specified')
    #             return None, None, None, None, None, None
    #     else:
    #         raise Exception('No stock specified')
    #         return None, None, None, None, None, None
            

    def case_6(self):
        '''
        Products and components have independent lifetimes. Potential failure of the componentscomponents is not included in the lifetime of the product. 
        Components can be replaced and/or reused. If the product fails but the status of the components is still good, the components can be reused in another product. 
        If the components fails but the product is still in good shape, a new components can be used to make the product operational again. Since the lifetimes of the 
        product and the components are independent, replacing the components should have no effect on the product's lifetime. Similarly, a components that is reused in 
        another product keeps its age and lifetime. Such a model would probably use parameters and transfer coefficients to determine the share of products and 
        components that will be reused depending on their age.

        Assuming that all components that flow out in the vehicles are reusable would lead to an overestimation of the amount of components that are actually
        fit to be reused. Instead, a logic is needed here to make a reasonable assumption. We reuse only the share of components that would be still be 
        usable according to their own technical lifetime. This logic can be further developed to use only the share of components that will be still useful
        for a specified minimum amount of time tau. The same logic is applied to the amount of products that is eligible to get a components replacement.

        As a first priority, vehicles will get a replacement from reused components. If these are not enough, new components fill the gap. 
        '''
        if self.s_pr is not None:
            if self.lt_pr is not None: 
                if self.lt_cm is not None:
                    if self.tau_pr is not None:
                        if self.tau_pr is not None:
                            self.sc_pr = np.zeros((len(self.t), len(self.t)))
                            self.oc_pr = np.zeros((len(self.t), len(self.t)))
                            self.i_pr = np.zeros(len(self.t))
                            self.sc_cm = np.zeros((len(self.t), len(self.t)))
                            self.oc_cm = np.zeros((len(self.t), len(self.t)))
                            self.i_cm = np.zeros(len(self.t))
                            self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                            self.o_cm = np.zeros(len(self.t))
                            self.o_pr = np.zeros(len(self.t))
                            replacement = np.zeros((len(self.t), len(self.t)))
                            reuse = np.zeros((len(self.t), len(self.t)))
                            # Initializing values
                            self.sc_pr[0,0] = self.s_pr[0]
                            self.o_pr[0] = 0 
                            self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
                            self.o_pr[1] = 0
                            self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
                            self.sc_pr[1,1] = self.i_pr[1]

                            self.sc_cm[0,0] = self.s_pr[0]
                            self.o_cm[0] = 0 
                            self.i_cm[0] = self.ds_pr[0] - self.o_pr[0]
                            self.o_cm[1] = 0
                            self.i_cm[1] = self.ds_pr[1] - self.o_pr[1]
                            self.sc_cm[1,1] = self.i_pr[1]
                            # construct the sf of a product of cohort tc remaining in the stock in year t
                            self.compute_sf_pr_tau() # Computes sf if not present already.
                            self.compute_sf_cm_tau() # Computes sf od component if not present already.
                            # all other years:            
                            for m in range(1, len(self.t)):  # for all years m, starting in second year
                                for c in range(m+1):
                                # 1) Compute outflow from previous age-cohorts up to m-1
                                    if  self.sf_cm[m-1,c] != 0 and self.sf_pr[m-1,c] != 0: # Else, inflow is 0.
                                        self.oc_pr[m, c] = self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c]))  # alculating outflows attributed to product failures
                                        self.oc_cm[m, c] = (self.sc_cm[m-1, c]- self.oc_pr[m, c])/self.sf_cm[m-1,c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))# Calculating outflows attributed to component failures 
                                        # Defining the amount of products eligible for component replacement 
                                        replacement[m,c] = self.sf_pr[m+self.tau_pr, c] 
                                        reuse[m,c] = self.oc_pr[m,c] * self.sf_cm[m+self.tau_cm, c]
                                        # Correcting outflows
                                        self.oc_pr[m, c] = self.oc_pr[m, c]+ self.oc_cm[m, c]  *(1- replacement[m,c])
                                        # The component outflows are equal to the component failures plus the product failures minus the amount of components reused
                                        self.oc_cm[m, c] =self.oc_cm[m, c] + (self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c]))) -  reuse[m,c]
                                        self.sc_pr[m,c] = self.sc_pr[m-1,c] - self.oc_pr[m, c]  # Computing real stock
                                        self.sc_cm[m,c] = self.sc_cm[m-1,c] - self.oc_cm[m,c]
                                    self.i_pr[m] = self.ds_pr[m] + self.oc_pr.sum(axis=1)[m] 
                                    self.i_cm[m] = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
                                    self.sc_pr[m,m] = self.i_pr[m]
                                    self.sc_cm[m,m] = self.i_cm[m]
                            #return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.oc_pr, self.oc_cm
                        else:
                            raise Exception('No component delay specified')
                            return None, None, None, None, None, None
                    else:
                        raise Exception('No product delay specified')
                        return None, None, None, None, None, None
                else:
                    raise Exception('No component lifetime specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No product lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None

    # This code is not needed if we find that the version above is correct
    # def case_6(self):
    #     '''
    #     Products and components have independent lifetimes. Potential failure of the componentscomponents is not included in the lifetime of the product. 
    #     Components can be replaced and/or reused. If the product fails but the status of the components is still good, the components can be reused in another product. 
    #     If the components fails but the product is still in good shape, a new components can be used to make the product operational again. Since the lifetimes of the 
    #     product and the components are independent, replacing the components should have no effect on the product's lifetime. Similarly, a components that is reused in 
    #     another product keeps its age and lifetime. Such a model would probably use parameters and transfer coefficients to determine the share of products and 
    #     components that will be reused depending on their age.

    #     Assuming that all components that flow out in the vehicles are reusable would lead to an overestimation of the amount of components that are actually
    #     fit to be reused. Instead, a logic is needed here to make a reasonable assumption. We reuse only the share of components that would be still be 
    #     usable according to their own technical lifetime. This logic can be further developed to use only the share of components that will be still useful
    #     for a specified minimum amount of time tau. The same logic is applied to the amount of products that is eligible to get a components replacement.

    #     As a first priority, vehicles will get a replacement from reused components. If these are not enough, new components fill the gap. 
    #     '''
    #     if self.s_pr is not None:
    #         if self.lt_pr is not None: 
    #             if self.lt_cm is not None:
    #                 if self.tau_pr is not None:
    #                     if self.tau_pr is not None:
    #                         self.sc_pr = np.zeros((len(self.t), len(self.t)))
    #                         self.oc_pr = np.zeros((len(self.t), len(self.t)))
    #                         self.i_pr = np.zeros(len(self.t))
    #                         self.sc_cm = np.zeros((len(self.t), len(self.t)))
    #                         self.oc_cm = np.zeros((len(self.t), len(self.t)))
    #                         self.i_cm = np.zeros(len(self.t))
    #                         self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
    #                         self.o_cm = np.zeros(len(self.t))
    #                         self.o_pr = np.zeros(len(self.t))
    #                         replacement = np.zeros((len(self.t), len(self.t)))
    #                         reuse = np.zeros((len(self.t), len(self.t)))
    #                         # Initializing values
    #                         self.sc_pr[0,0] = self.s_pr[0]
    #                         self.o_pr[0] = 0 
    #                         self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
    #                         self.o_pr[1] = 0
    #                         self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
    #                         self.sc_pr[1,1] = self.i_pr[1]

    #                         self.sc_cm[0,0] = self.s_pr[0]
    #                         self.o_cm[0] = 0 
    #                         self.i_cm[0] = self.ds_pr[0] - self.o_pr[0]
    #                         self.o_cm[1] = 0
    #                         self.i_cm[1] = self.ds_pr[1] - self.o_pr[1]
    #                         self.sc_cm[1,1] = self.i_pr[1]
    #                         # construct the sf of a product of cohort tc remaining in the stock in year t
    #                         self.compute_sf_pr_tau() # Computes sf if not present already.
    #                         self.compute_sf_cm_tau() # Computes sf od component if not present already.
    #                         # all other years:            
    #                         for m in range(1, len(self.t)):  # for all years m, starting in second year
    #                             for c in range(m+1):
    #                             # 1) Compute outflow from previous age-cohorts up to m-1
    #                                 if  self.sf_cm[m-1,c] != 0 and self.sf_pr[m-1,c] != 0: # Else, inflow is 0.
    #                                     self.oc_pr[m, c] = self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c]))  # alculating outflows attributed to product failures
    #                                     self.oc_cm[m, c] = (self.sc_pr[m-1, c]- self.oc_pr[m, c])/self.sf_cm[m-1,c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))# Calculating outflows attributed to component failures 
    #                                     # Defining the amount of products eligible for component replacement 
    #                                     replacement[m,c] = self.sf_pr[m+self.tau_pr, c] 
    #                                     reuse[m,c] = self.sf_cm[m+self.tau_cm, c]
    #                                     # Correcting outflows
    #                                     self.oc_pr[m, c] = self.oc_pr[m, c]+ self.oc_cm[m, c]  *(1- replacement[m,c])
    #                                     self.oc_cm[m, c] = self.oc_cm[m, c] + self.sc_pr[m-1, c] /self.sf_pr[m-1,c] * abs((self.sf_pr[m, c] - self.sf_pr[m-1, c])) - self.oc_pr[m, c] * reuse[m,c]
    #                                     self.sc_pr[m,c] = self.sc_pr[m-1,c] - self.oc_pr[m, c]  # Computing real stock
    #                                     self.sc_cm[m,c] = self.sc_cm[m-1,c] - self.oc_cm[m,c]
    #                                 self.i_pr[m] = self.ds_pr[m] + self.oc_pr.sum(axis=1)[m] 
    #                                 self.i_cm[m] = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
    #                                 self.sc_pr[m,m] = self.i_pr[m]
    #                             # TODO: Need to add stock by cohort of batteries

    #                         return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.oc_pr, self.oc_cm
    #                     else:
    #                         raise Exception('No component delay specified')
    #                         return None, None, None, None, None, None
    #                 else:
    #                     raise Exception('No product delay specified')
    #                     return None, None, None, None, None, None
    #             else:
    #                 raise Exception('No component lifetime specified')
    #                 return None, None, None, None, None, None
    #         else:
    #             raise Exception('No product lifetime specified')
    #             return None, None, None, None, None, None
    #     else:
    #         raise Exception('No stock specified')
    #         return None, None, None, None, None, None
            
    def case_7(self):
        '''
        The model does not use lifetimes at all. It would make sense if the inflows of products and components are calculated as a percentage of 
        the stock (Stock and birth rate model in the paper Lauinger et al.). OK for models where the stock is stable, not suited for studying the 
        penetration of new products and technologies. 1 product = 1 component, no replacements allowed

        This type of models can be driven by either a death or a birth rate. The component outflows and the product outflows are equivalent and 
        the rate encompases failures in both products. 

        This case computes model using death rate
        '''
        if self.s_pr is not None:
            if self.d is not None: 
                self.sc_pr = np.zeros((len(self.t), len(self.t)))
                self.o_pr = np.zeros( len(self.t))
                self.i_pr = np.zeros(len(self.t))
                self.sc_cm = np.zeros((len(self.t), len(self.t)))
                self.o_cm = np.zeros(len(self.t))
                self.i_cm = np.zeros(len(self.t))
                self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                # construct the sf of a product of cohort tc remaining in the stock in year t
                # all other years:
                for m in range(1, len(self.t)):  # for all years m, starting in second year
                    # 1) Since leaching approach does not track the cohorts, we cannot determine oc
                    self.o_pr[m] = self.s_pr[m] * self.d # Outflows are a death fraction of the stock 
                    self.o_cm[m] = self.o_pr[m] # Components die with products      
                    self.i_pr[m] = self.ds_pr[m] + self.o_pr[m]
                    self.i_cm[m] = self.i_pr[m]
                    # 3) Add new inflow to stock and determine future decay of new age-cohort
                    self.sc_pr[m, m] = self.i_pr[m]
                # 4) Determining the values for the component
                self.sc_cm = self.sc_pr
                return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.o_cm
            else:
                # No death rate specified
                raise Exception('No death rate specified')
                return None, None, None, None, None, None
        else:
                # No stock specified
                raise Exception('No stock specified')
                return None, None, None, None, None, None
                
    def case_7b(self):
        '''
        The model does not use lifetimes at all. It would make sense if the inflows of products and components are calculated as a percentage of 
        the stock (Stock and birth rate model in the paper Lauinger et al.). OK for models where the stock is stable, not suited for studying the 
        penetration of new products and technologies. 1 product = 1 component, no replacements allowed

        This type of models can be driven by either a death or a birth rate. The component outflows and the product outflows are equivalent and 
        the rate encompases failures in both products. 

        This case computes model using birth rate
        '''
        if self.s_pr is not None:
            if self.b is not None: 
                self.sc_pr = np.zeros((len(self.t), len(self.t)))
                self.o_pr = np.zeros( len(self.t))
                self.i_pr = np.zeros(len(self.t))
                self.sc_cm = np.zeros((len(self.t), len(self.t)))
                self.o_cm = np.zeros(len(self.t))
                self.i_cm = np.zeros(len(self.t))
                self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                # construct the sf of a product of cohort tc remaining in the stock in year t
                # all other years:
                for m in range(1, len(self.t)):  # for all years m, starting in second year
                    # 1) Since leaching approach does not track the cohorts, we cannot determine oc
                    self.i_pr[m] = self.s_pr[m] * self.b # Outflows are a death fraction of the stock 
                    self.i_cm[m] = self.i_pr[m] # Components die with products      
                    self.o_pr[m] = self.i_pr[m] - self.ds_pr[m]
                    self.o_cm[m] = self.o_pr[m]
                    # 3) Add new inflow to stock and determine future decay of new age-cohort
                    self.sc_pr[m, m] = self.i_pr[m]
                # 4) Determining the values for the component
                self.sc_cm = self.sc_pr
                return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.o_cm
            else:
                # No birth rate specified
                raise Exception('No birth rate specified')
                return None, None, None, None, None, None
        else:
                # No stock specified
                raise Exception('No stock specified')
                return None, None, None, None, None, None

    def case_8(self):
        '''
        The model does not use lifetimes at all. It would make sense if the inflows of cars and batteries are calculated as a percentage of the stock 
        (Stock and birth rate model in the paper with Dirk). OK for models where the stock is stable, not suited for studying the penetration of new 
        products and technologies.

        This case computes model using death rate
        '''
        if self.s_pr is not None:
            if self.d is not None: 
                if self.r is not None:
                    self.sc_pr = np.zeros((len(self.t), len(self.t)))
                    self.o_pr = np.zeros( len(self.t))
                    self.i_pr = np.zeros(len(self.t))
                    self.sc_cm = np.zeros((len(self.t), len(self.t)))
                    self.o_cm = np.zeros(len(self.t))
                    self.i_cm = np.zeros(len(self.t))
                    self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                    # construct the sf of a product of cohort tc remaining in the stock in year t
                    # all other years:
                    for m in range(1, len(self.t)):  # for all years m, starting in second year
                        # 1) Since leaching approach does not track the cohorts, we cannot determine oc
                        self.o_pr[m] = self.s_pr[m] * self.d # Outflows are a death fraction of the stock 
                        self.o_cm[m] = self.o_pr[m] * self.r # Components die with products      
                        self.i_pr[m] = self.ds_pr[m] + self.o_pr[m]
                        self.i_cm[m] = self.ds_pr[m] + self.o_cm[m]
                        # 3) Add new inflow to stock and determine future decay of new age-cohort
                        self.sc_pr[m, m] = self.i_pr[m]
                    # 4) Determining the values for the component
                    self.sc_cm = self.sc_pr
                    return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.o_cm
                else:
                    # No replacement rate specified
                    raise Exception('No replacement rate specified')
                    return None, None, None, None, None, None
            else:
                # No death rate specified
                raise Exception('No death rate specified')
                return None, None, None, None, None, None
        else:
                # No stock specified
                raise Exception('No stock specified')
                return None, None, None, None, None, None
                
    def case_9(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. No component replacement, when it dies, the product dies.  
        There should be some extra outflows of products as well (accidents), which could be modelled by a stock and death rate approach. 
        1 product = one component, Outflow component = outflow product.  
        '''
        if self.s_pr is not None:
            if self.lt_cm is not None:
                if self.d is not None:
                    self.sc_pr = np.zeros((len(self.t), len(self.t)))
                    self.oc_pr = np.zeros((len(self.t), len(self.t)))
                    self.i_pr = np.zeros(len(self.t))
                    self.sc_cm = np.zeros((len(self.t), len(self.t)))
                    self.oc_cm = np.zeros((len(self.t), len(self.t)))
                    self.i_cm = np.zeros(len(self.t))
                    self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                    self.o_cm = np.zeros(len(self.t))
                    self.o_pr = np.zeros(len(self.t))
                    # construct the sf of a product of cohort tc remaining in the stock in year t
                    self.compute_sf_cm() # Computes sf if not present already.
                    for m in range(1, len(self.t)): 
                        for c in range(m+1): 
                            # Calculate the outflows related to component lifetime
                            self.oc_cm[m,c]    = self.i_cm[c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))
                            # Calculate outflows due to accidents/pruduct failures
                            self.oc_pr[m,c]         = self.i_cm[c] * self.sf_cm[m,c] * self.d
                            # Updating value of the stock
                            self.sc_cm[m,c]      = self.i_cm[c] * self.sf_cm[m,c]
                        # Calculate the inflows related to these outflows
                        self.i_cm[m]          = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
                        # Calculate total inflows
                        self.i_pr[m]      = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m] + self.oc_pr.sum(axis=1)[m]
                    self.oc_pr         = self.oc_pr + self.oc_cm
                    self.oc_cm         = self.oc_pr
                    self.i_cm          = self.i_pr
                    self.sc_pr         = self.sc_cm
                    return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.oc_cm
                else:
                    raise Exception('No product death rate specified')
                return None, None, None, None, None, None
            else:
                raise Exception('No lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None

    def case_10(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. No component replacement, when it dies, the product dies, 
        but some components from failed products can be reused in new products.  There should be some extra outflows of products as well (accidents), 
        which could be modelled by a stock and death rate approach. Outflow component <= outflow product. This case is probably not the most meaningful.

        In this case some of the components are deemed to be reusable. This makes sense if we consider that some of the outflows are due to product failures. 
        We can here consider then that only a fraction of the components that would qualify can actually be reused. We do this by defining a reuse rate of the
        components in failed products.
        '''
        if self.s_pr is not None:
            if self.lt_cm is not None:
                if self.d is not None:
                    if self.r is not None: 
                        self.sc_pr = np.zeros((len(self.t), len(self.t)))
                        self.oc_pr = np.zeros((len(self.t), len(self.t)))
                        self.i_pr = np.zeros(len(self.t))
                        self.sc_cm = np.zeros((len(self.t), len(self.t)))
                        self.oc_cm = np.zeros((len(self.t), len(self.t)))
                        self.i_cm = np.zeros(len(self.t))
                        self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                        self.o_cm = np.zeros(len(self.t))
                        self.o_pr = np.zeros(len(self.t))
                        reuse = np.zeros((len(self.t), len(self.t)))
                        # construct the sf of a product of cohort tc remaining in the stock in year t
                        self.compute_sf_cm() # Computes sf if not present already.
                        for m in range(1, len(self.t)): 
                            for c in range(m+1): 
                                # Calculate the outflows related to component lifetime minus the reused share
                                self.oc_cm[m,c]    = self.i_cm[c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))
                                # Calculate outflows due to accidents/pruduct failures
                                self.oc_pr[m,c]    = self.i_cm[c] * self.sf_cm[m,c] * self.d
                                # Defining share of batteries that can be reused from the death rate
                                reuse[m,c]         = self.oc_pr[m,c] * self.r
                                # Updating value of the stock
                                self.sc_cm[m,c]    = self.i_cm[m] * self.sf_cm[m,c]
                            # Calculate the inflows related to these outflows
                            self.i_cm[m]          = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m] 
                            # Calculate total inflows
                            self.i_pr[m]      = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m] + self.oc_pr.sum(axis=1)[m]
                        self.oc_pr         = self.oc_pr + self.oc_cm 
                        # Subtracting the components that have been reused from the in- and outflows
                        self.oc_cm         = self.oc_pr - reuse
                        self.i_cm          = self.i_pr - reuse.sum(axis=1)
                        #self.sc_pr         = self.sc_cm
                        return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.oc_cm
                    else:
                        raise Exception('No replacement rate specified')
                        return None, None, None, None, None, None
                else:
                    raise Exception('No death rate specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No component lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None

    def case_11(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. 
        The component can be replaced, but old components cannot be reused. 
        This model assumes that a product would live infinitely as long as the component keeps being replaced.
        There should be some extra outflows of products as well (accidents), which could be modelled by a stock and death rate approach. Outflow component >= outflow product.  
        '''
        if self.s_pr is not None:
            if self.lt_cm is not None:
                if self.d is not None:
                    self.sc_pr = np.zeros((len(self.t), len(self.t)))
                    self.oc_pr = np.zeros((len(self.t), len(self.t)))
                    self.i_pr = np.zeros(len(self.t))
                    self.sc_cm = np.zeros((len(self.t), len(self.t)))
                    self.oc_cm = np.zeros((len(self.t), len(self.t)))
                    self.i_cm = np.zeros(len(self.t))
                    self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                    self.o_cm = np.zeros(len(self.t))
                    self.o_pr = np.zeros(len(self.t))
                    reuse = np.zeros((len(self.t), len(self.t)))
                    # Initializing values
                    self.sc_pr[0,0] = self.s_pr[0]
                    self.o_pr[0] = 0 
                    self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
                    self.o_pr[1] = 0
                    self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
                    self.sc_pr[1,1] = self.i_pr[1]

                    # construct the sf of a product of cohort tc remaining in the stock in year t
                    self.compute_sf_cm() # Computes sf if not present already.
                    for m in range(1, len(self.t)): 
                        for c in range(m+1): 
                            self.oc_cm[m,c]    = self.i_cm[c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c]))
                            # the outflow of batteries from a cohort is equal to the outflows of the vehicles
                            self.o_pr[m]         = self.s_pr[m] * self.d
                            self.sc_cm[m,c]      = self.i_cm[m] * self.sf_cm[m,c]
                            self.sc_pr[m,c]      = self.sc_cm[m,c] 
                        self.i_pr[m]          = self.ds_pr[m] + self.o_pr[m] 
                        self.i_cm[m]          = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]

                    return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.oc_cm
                else:
                    raise Exception('No death rate specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No component lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None

    def case_12(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. 
        The component can be replaced, and old components can be reused. 
        This model assumes that a product would live infinitely as long as the component keeps being replaced. 
        There should be some extra outflows of products as well (accidents), which could be modelled by a stock and death rate approach. 
        Outflow component <= or >= outflow product depending on assumptions for reuse and replacement.

        We take a similar approach to case 6 here, only the products do not have a lifetime and therefore are always elegible to get a component replacement. 
        The rules for when a component can be reused become more tricky because we now assume that pretty much all outflows are because of component failured. 
        So what we do is take the share of components that is exiting the fleet due to product crashes and take a share of them for reuse. 
        Since there is a leaching approach, we do not know the cohort of the products being crashed and therefore we have to assume that the components will be fit for reuse for a full component lifetime. 
        '''
        if self.s_pr is not None:
            if self.lt_cm is not None:
                if self.d is not None:
                    if self.r is not None: 
                        self.sc_pr = np.zeros((len(self.t), len(self.t)))
                        self.oc_pr = np.zeros((len(self.t), len(self.t)))
                        self.i_pr = np.zeros(len(self.t))
                        self.sc_cm = np.zeros((len(self.t), len(self.t)))
                        self.oc_cm = np.zeros((len(self.t), len(self.t)))
                        self.i_cm = np.zeros(len(self.t))
                        self.ds_pr = np.concatenate((np.array([0]), np.diff(self.s_pr)))
                        self.o_cm = np.zeros(len(self.t))
                        self.o_pr = np.zeros(len(self.t))
                        reuse = np.zeros( len(self.t))
                        # Initializing values
                        self.sc_pr[0,0] = self.s_pr[0]
                        self.o_pr[0] = 0 
                        self.i_pr[0] = self.ds_pr[0] - self.o_pr[0]
                        self.o_pr[1] = 0
                        self.i_pr[1] = self.ds_pr[1] - self.o_pr[1]
                        self.sc_pr[1,1] = self.i_pr[1]

                        # construct the sf of a product of cohort tc remaining in the stock in year t
                        self.compute_sf_cm() # Computes sf if not present already.
                        for m in range(1, len(self.t)): 
                            for c in range(m+1): 
                                self.oc_cm[m,c]    = self.i_cm[c] * abs((self.sf_cm[m, c] - self.sf_cm[m-1, c])) #- reuse[m,c]
                                # the outflow of components from a cohort is equal to the outflows of the vehicles
                                self.o_pr[m]         = self.s_pr[m] * self.d
                                self.sc_cm[m,c]      = self.i_cm[m] * self.sf_cm[m,c]
                                self.sc_pr[m,c]      = self.sc_cm[m,c] 
                            self.i_pr[m]          = self.ds_pr[m] + self.o_pr[m] 
                            self.i_cm[m]          = self.ds_pr[m] + self.oc_cm.sum(axis=1)[m]
                        # Define sahre of reusable components
                        reuse         = self.o_pr * self.r
                        # Fix outflows because of reuse
                        self.oc_cm = self.oc_cm - np.diag(reuse)
                        self.i_cm = self.i_cm - reuse
                        # We need to correct for negative outflows in case there is an excess of batteries for reuse                        
                        return self.sc_pr, self.sc_cm, self.i_pr, self.i_cm, self.o_pr, self.oc_cm
                    else:
                        raise Exception('No reuse rate specified')
                        return None, None, None, None, None, None
                else:
                    raise Exception('No death rate specified')
                    return None, None, None, None, None, None
            else:
                raise Exception('No component lifetime specified')
                return None, None, None, None, None, None
        else:
            raise Exception('No stock specified')
            return None, None, None, None, None, None
