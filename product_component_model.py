"""
Class ProductComponentModel
Check https://github.com/fernaag/product_component_model for latest version.

Methods for handling product-component interactions under different assumptions. 

Created on Mon Jul 05 2021

@authors: Fernando Aguilar Lopez & Romain Billy, NTNU Trondheim, Norway.
Built on the previous work done by Stephan Pauliuk in the dynamic_stock_model. 

standard abbreviation: PCM or pcm 

dependencies: #TODO Update this
    numpy >= 1.9
    scipy >= 0.14

Repository for this class, documentation, and tutorials: 
https://github.com/fernaag/product_component_model

"""

import numpy as np
import scipy.stats

def __version__():
    """Return a brief version string and statement for this class."""
    return str('1.0'), str(
    """Class ProductComponentModel, pcm. Version 1.0. 
    Last change: Thu Dec 2nd, 2021. 
    Check https://github.com/fernaag/product_component_model for latest version.
    """)

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
    
    s_tpc : stock broken down by time, product cohort, and component cohort
    
    s_pr : Discrete time series for product stock, total
    s_cm : Discrete time series for component stock, total

    ds_pr : Discrete time series for product stock change, total
    ds_cm : Discrete time series for component stock change, total

    lt_pr : lifetime distribution of product: dictionary
    lt_cm : lifetime distribution of component: dictionary

    replacement_coeff: Replacement rate of component
    reuse_coeff: share of components from failed products that can be reused

    d: Death rate 
    b: Birth rate

    pdf_pr: probability density function of product, distribution of outflow from a specific age-cohort
    pdf_cm: probability density function of component, distribution of outflow from a specific age-cohort
    
    sf_pr: survival function for different age-cohorts of product, year x age-cohort table
    sf_cm: survival function for different age-cohorts of component, year x age-cohort table
    
    hz_pr: hazard function for different age-cohorts of product, year x age-cohort table
    hz_cm: hazard function for different age-cohorts of component, year x age-cohort table


    name : string, optional
        Name of the product component model, default is 'PCM'
    """

    """
    Basic initialisation and dimension check methods
    """

    def __init__(self, t=None, i_pr=None, i_cm=None, o_pr=None, o_cm=None, 
                 s_pr=None, s_cm=None, lt_pr=None, lt_cm=None, sc_pr=None, 
                 sc_cm=None, oc_pr=None, oc_cm=None, name='PCM', pdf_pr=None, 
                 pdf_cm=None, sf_pr=None, sf_cm=None, reuse_coeff=None,
                 replacement_coeff=None, tau_cm=None, tau_pr=None, d=None, b=None, 
                 hz_cm=None, hz_pr = None):
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

        self.reuse_coeff = reuse_coeff # optional
        self.replacement_coeff = replacement_coeff # optional
        
        self.tau_cm = tau_cm # optional
        self.tau_pr = tau_pr # optional

        self.d = d # optional
        self.b = b # optional

        self.pdf_pr = pdf_pr # optional
        self.sf_pr  = sf_pr # optional

        self.pdf_cm = pdf_cm # optional
        self.sf_cm  = sf_cm # optional
        self.hz_cm = hz_cm # optional
        self.hz_pr = hz_pr # optional
        

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
        Survival table self.sf(m,n) denotes the share of a product inflow in year n (age-cohort) still present at the end of year m (after m-n years).
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
        Survival table self.sf(m,n) denotes the share of a component inflow in year n (age-cohort) still present at the end of year m (after m-n years).
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

    def compute_hz_pr(self): # hazard functions
        """
        Hazard table self.hz_pr(m,n) denotes the probability of failing during year m 
        for a product inflow from year n (age-cohort) 
        still present at the beginning of year m (after m-n years).
        """
        self.hz_pr = np.zeros((len(self.t), len(self.t)))
        if self.sf_pr is None:
            print("survival function not defined")
        else:
            for n in range(len(self.t)): # for each cohort n
                # calculating the hazard function for the first year
                self.hz_pr[n,n] = 1-self.sf_pr[n,n] #sf = 1 at the beginning of the first year
                for m in range(n, len(self.t)-1): # for all years > n, stops at t-1 because we calculate hz(m+1)
                    if self.sf_pr[m,n] != 0: # if sf = 0, hz = 1 (avoids dividing by 0)
                        self.hz_pr[m+1,n] = (self.sf_pr[m,n] - self.sf_pr[m+1,n]) / self.sf_pr[m,n]
                    else:
                        self.hz_pr[m+1,n] = 1
        return self.hz_pr

    def compute_hz_cm(self): # hazard functions
        """
        Hazard table self.hz_cm(m,n) denotes the probability of failing during year m 
        for a component inflow from year n (age-cohort) 
        still present at the beginning of year m (after m-n years).
        """
        self.hz_cm = np.zeros((len(self.t), len(self.t)))
        if self.sf_cm is None:
            print("survival function not defined")
        else:
            for n in range(len(self.t)): # for each cohort n
                # calculating the hazard function for the first year
                self.hz_cm[n,n] = 1-self.sf_cm[n,n] # sf = 1 at the beginning of the first year
                for m in range(n, len(self.t)-1): # for all years > n, stops at t-1 because we calculate hz(m+1)
                    if self.sf_cm[m,n] != 0: # if sf = 0, hz = 1 (avoids dividing by 0)
                        self.hz_cm[m+1,n] = (self.sf_cm[m,n] - self.sf_cm[m+1,n]) / self.sf_cm[m,n]
                    else:
                        self.hz_cm[m+1,n] = 1 
        return self.hz_cm

    def compute_hz_cm_old(self): # hazard functions
        """
        Hazard table self.hz_cm(m,n) denotes the probability of failing during year m 
        for a component inflow from year n (age-cohort) 
        still present at the beginning of year m (after m-n years).
        """
        self.hz_cm = np.zeros((len(self.t), len(self.t)))
        if self.sf_cm is None:
            print("survival function not defined")
        else:
            for m in range(0, len(self.t)-1): 
                self.hz_cm[m,m] = 1-self.sf_cm[m,m] #sf is equal to 1 at the beginning of the first year

                self.hz_cm[m+1:,m] = (self.sf_cm[m:-1,m] - self.sf_cm[m+1::,m]) / self.sf_cm[m:-1,m] 

        return self.hz_cm

    
    def case_1(self):
        '''
        Products have a lifetime that includes all types of failures, component EoL being one of them.
        No component replacement. 1 product = one component, Outflow component = outflow product 
        Since this case includes all possible failures, the choice of lifetime should not be limted 
        to the technical lifetime of the product itself but should also consider how this might be 
        affected by failures in the component and potential damages. 
        Therefore, the probability that a failure occurs is greater than the techical lifetime 
        of the product itself and should lead to a shorter choice of lifetime. 
        This approach is equivalent as the traditional one-lifetime approach used in dynamic models.
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is  None: 
            raise Exception('No product lifetime specified')

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
            # outflow table is filled row-wise, for each year m.
            self.oc_pr[m, 0:m] = self.sc_pr[m-1, 0:m] - self.sc_pr[m, 0:m] 
            # 2) Determine inflow from mass balance:
            
            if self.sf_pr[m,m] != 0: # Else, inflow is 0.
                # allow for outflow during first year by rescaling with 1/sf[m,m]
                self.i_pr[m] = (self.s_pr[m] - self.sc_pr[m, :].sum()) / self.sf_pr[m,m]
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
                
          
          
    def case_1_hz(self):
        '''
         Products have a lifetime that includes all types of failures, component EoL being one of them.
         No component replacement. 1 product = one component, Outflow component = outflow product
         Since this case includes all possible failures, the choice of lifetime 
         should not be limted to the technical lifetime of the product itself 
         but should also consider how this might be affected by failures in the component 
         and potential damages. 
         Therefore, the probability that a failure occurs is greater than the techical lifetime
         of the product itself and should lead to a shorter choice of lifetime.         
         In this version, all dynamics are computed using the hazard function 
         instead of the survival curve. 

         '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is  None: 
            raise Exception('No product lifetime specified')

        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous (same year) product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by component failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        
        # construct the hazard functions of product and components
        self.compute_sf_pr() # Computes sf if not present already.
        self.compute_hz_pr() # product hazard function
       
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]
                            
        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                # Probability of any failure is calculated using product hazard function 
                self.o_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] * self.hz_pr[m,:m] 
                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                # Add new cohort to stock
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
           
            # Calculate new inflow, accounting for outflows in the first year
            
            self.o_tpc[m,m,m] = self.s_tpc[m,m,m] * self.hz_pr[m,m] 
    
            self.i_pr[m] = self.ds_pr[m] + self.o_tpc[m,:,:].sum()                   
            self.i_cm[m] = self.i_pr[m] #one new component per new product       
                
        self.o_pr = np.einsum('tpc->t', self.o_tpc)
        self.o_cm = np.einsum('tpc->t', self.o_tpc)
        self.oc_pr = np.einsum('tpc->tp', self.o_tpc)
        self.oc_cm = np.einsum('tpc->tc', self.o_tpc)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)
          
    def case_2(self):
        '''
        Products have a lifetime that includes all types of failures, component EoL being one of them. 
        Still, it is assumed that some components will be replaced at a given rate replacement_coeff. 
        More than one component is used in the lifetime of the product.
        Outflow component >= outflow product
        In this case the failure of components is accounted for by a replacement rate
        and therefore the lifetime of the vehicle can be assumed to not be affected
        to the same extent as in case 1. 
        Limitations to this approach might be that the outflows and component inflows 
        are estimated at different times as the case where a separate lifetime 
        would be calculated for the components. 
        It is more suitable if the stock is approximately constant.  
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is None:
            raise Exception('No product lifetime specified')
        if self.replacement_coeff is None: 
            raise Exception('No replacement rate specified')
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
            # outflow table is filled row-wise, for each year m.
            self.oc_pr[m, 0:m] = self.sc_pr[m-1, 0:m] - self.sc_pr[m, 0:m] 
            # 2) Determine inflow from mass balance:
            
            if self.sf_pr[m,m] != 0: # Else, inflow is 0.
                # allow for outflow during first year by rescaling with 1/sf[m,m]
                self.i_pr[m] = (self.s_pr[m] - self.sc_pr[m, :].sum()) / self.sf_pr[m,m]
                                
            # 3) Add new inflow to stock and determine future decay of new age-cohort
            self.sc_pr[m::, m] = self.i_pr[m] * self.sf_pr[m::, m]
            self.oc_pr[m, m]   = self.i_pr[m] * (1 - self.sf_pr[m, m])
        # 4) Determining the values for the component
        self.s_cm = self.s_pr
        self.i_cm = self.i_pr * (1 + self.replacement_coeff)
        self.o_cm = self.i_cm - self.compute_stock_change_cm()

        # Calculating total values
        self.o_pr = self.oc_pr.sum(axis=1)


    def case_3(self):
        '''
        Products and components have independent hazard functions. 
        Potential failure of the component is not included in the hazard function of the product.
        Components can neither be replaced nor reused, meaning that if 
        either the product or the component fail, they are both scrapped.
        In this case, the combined hazard functions of the component and the product
        yield a smaller in-use time that dictates the outflows based on all failures. 
        This would be a similar case as 1, but the in-use time is the result of 
        the combined dynamics of the hazard functions.
        Double-counting is accounted for by considering the overlapping probability
        of failure of product and component during the same year.

        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is  None: 
            raise Exception('No product lifetime specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous (same year) product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by component failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        
        # construct the hazard functions of product and components
        self.compute_sf_pr() # Computes sf if not present already.
        self.compute_sf_cm() # Computes sf of component if not present already.
        self.compute_hz_pr() # product hazard function
        self.compute_hz_cm() # component hazard function
       
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]
                            
        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                # simultaneous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,:m,:m] = self.s_tpc[m-1,:m,:m] * self.hz_pr[m,:m] * self.hz_cm[m,:m]
                # failure from products only given by multiplying product hazard function and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,:m,:m] = self.s_tpc[m-1,:m,:m] * self.hz_pr[m,:m] * (1-self.hz_cm[m,:m])
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,:m,:m] = self.s_tpc[m-1,:m,:m] * self.hz_cm[m,:m] * (1-self.hz_pr[m,:m])

                # Total outflows for all cohorts <m
                self.o_tpc[m,:m,:m] = (self.o_tpc_pr[m,:m,:m] 
                                    + self.o_tpc_cm[m,:m,:m] 
                                    + self.o_tpc_both[m,:m,:m])
                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                # Add new cohort to stock
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
           
            # Calculate new inflow, accounting for outflows in the first year
            
            self.o_tpc_both[m,m,m] = self.s_tpc[m,m,m] * self.hz_pr[m,m] * self.hz_cm[m,m]
            # failure from products only given by multiplying product hazard function and
            # assuming that the probability of component failure is independent from product failure
            self.o_tpc_pr[m,m,m] = self.s_tpc[m,m,m] * self.hz_pr[m,m] * (1-self.hz_cm[m,m])
            # failure from components only given by multiplying component hazard function and
            # assuming that the probability of product failure is independent from product failure
            self.o_tpc_cm[m,m,m] = self.s_tpc[m,m,m] * self.hz_cm[m,m] * (1-self.hz_pr[m,m])
            # Total outflows
            self.o_tpc[m,m,m] = self.o_tpc_pr[m,m,m] + self.o_tpc_cm[m,m,m] + self.o_tpc_both[m,m,m]
    
            self.i_pr[m] = self.ds_pr[m] + self.o_tpc[m,:,:].sum()                   
            self.i_cm[m] = self.i_pr[m] #one new component per new product       
                
        self.o_pr = np.einsum('tpc->t', self.o_tpc)
        self.o_cm = np.einsum('tpc->t', self.o_tpc)
        self.oc_pr = np.einsum('tpc->tp', self.o_tpc)
        self.oc_cm = np.einsum('tpc->tc', self.o_tpc)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)
   
    def case_4(self, max_age_cm_reuse=None):
        '''
        Products and components have independent hazard functions.
        Potential failure of the component is not included in the hazard function of the product. 
        Components cannot be replaced, but they can be reused.
        If the component fails, the product is scrapped.         
        If the product fails but the status of the component is still good, 
        it can be reused in a new product according to the reuse_coeff. 
        In addition to the approach in case 4, 
        here the maximum age of the reused components may be defined as 
        an additional constraint to the reuse_coeff.
        The stock composition of the product is no longer equal 
        to the stock composition of the component, but the total stock is. 
        '''
        
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is  None: 
            raise Exception('No product lifetime specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.reuse_coeff is None:
            raise Exception('No reuse coefficient specified')
        
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        
        # component reuse (in new products)
        self.reuse_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
         
        # construct the hazard functions of product and components
        self.compute_sf_pr() # Computes sf if not present already.
        self.compute_sf_cm() # Computes sf of component if not present already.
        self.compute_hz_pr() # product hazard function
        self.compute_hz_cm() # component hazard function
 
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]
        new_inflow_pr = 0
                            
        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for p in range(m):     # for all product cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                    self.o_tpc_both[m,p,:m] = self.s_tpc[m-1,p,:m] * self.hz_pr[m,p] * self.hz_cm[m,:m]
                    # failure from products only given by multiplying product hazard function and
                    # assuming that the probability of component failure is independent from product failure
                    self.o_tpc_pr[m,p,:m] = self.s_tpc[m-1,p,:m] * self.hz_pr[m,p] * (1-self.hz_cm[m,:m])
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of product failure is independent from product failure
                    self.o_tpc_cm[m,p,:m] = (self.s_tpc[m-1,p,:m]
                                          * self.hz_cm[m,:m] 
                                          * (1-self.hz_pr[m,p]))
                
                # Total outflows for all cohorts <m
                self.o_tpc[m,:m,:m] = self.o_tpc_pr[m,:m,:m] + self.o_tpc_cm[m,:m,:m] + self.o_tpc_both[m,:m,:m]
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                               
                # component reuse (in new products)
                if max_age_cm_reuse is None:
                    max_age_cm_reuse = len(self.t)
                oldest_cohort = max(m - max_age_cm_reuse, 0)
                self.reuse_tpc_cm[m,:m,oldest_cohort:m] = (self.reuse_coeff
                                                        * self.o_tpc_pr[m,:m,oldest_cohort:m]) 
                
                # calculate the size of the product demand for year m
                demand_pr = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()

                # if the demand is smaller than the potential for component reuse,
                # the oldest components will not be reused
                components_for_reuse =  np.einsum('pc -> c', 
                                                 self.reuse_tpc_cm[m,:m,oldest_cohort:m])
                c = oldest_cohort
                while (components_for_reuse.sum() > demand_pr) and (c < m):
                    if components_for_reuse[c+1:].sum() > demand_pr:
                        components_for_reuse[c] = 0
                    else:
                        components_for_reuse[c] = demand_pr - components_for_reuse[c+1:].sum()
                    c += 1       

                # Add reused components to stock in a new product cohort
                self.s_tpc[m,m,:m] = components_for_reuse
                
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m].sum()
           
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_pr[m,m] * self.hz_cm[m,c]
                # failure from products only given by multiplying product hazard function and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_pr[m,m] * (1-self.hz_cm[m,m])
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.hz_pr[m,m])

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = (self.o_tpc_pr[m,m,:m+1]
                                 + self.o_tpc_cm[m,m,:m+1] 
                                 + self.o_tpc_both[m,m,:m+1])

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 
            
        # Calculating aggregated values      
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.oc_pr  = np.einsum('tpc->tp', self.o_tpc)
        self.oc_cm  = np.einsum('tpc->tc', self.o_tpc - self.reuse_tpc_cm)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)
           
        
    def case_5(self):
        '''
        Products and components have independent hazard functions.
        Potential failure of the component is not included in the hazard function of the product. 
        Components cannot be reused, but they can be replaced. 
        If the component fails, the product is can get a replacement according to the replacement_coeff. 
        The stock composition of the product is no longer equal to the stock composition of the component,
        but the total stock is. 
        '''
        
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is  None: 
            raise Exception('No product lifetime specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.replacement_coeff is None:
            raise Exception('No replacement coefficient specified')
        
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        
        # product reuse
        self.replacement_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        
        # construct the hazard functions of product and components
        self.compute_sf_pr() # Computes sf if not present already.
        self.compute_sf_cm() # Computes sf of component if not present already.
        self.compute_hz_pr() # product hazard function
        self.compute_hz_cm() # component hazard function
 
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]
        new_inflow_pr = 0

        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for c in range(m):     # for all component cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions
                    # for all cohorts <m
                    self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_pr[m,:m] * self.hz_cm[m,c]
                    # failure from products only given by multiplying product hazard function and
                    # assuming that the probability of component failure 
                    # is independent from product failure
                    self.o_tpc_pr[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_pr[m,:m] * (1-self.hz_cm[m,c])
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of product failure 
                    # is independent from product failure
                    self.o_tpc_cm[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,c] * (1-self.hz_pr[m,:m])
                
                # Total outflows for all cohorts <m
                self.o_tpc[m,:m,:m] = (self.o_tpc_pr[m,:m,:m]
                                    + self.o_tpc_cm[m,:m,:m] 
                                    + self.o_tpc_both[m,:m,:m])
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                
                # component replacement (in old products)
                self.replacement_tpc_cm[m,:m,:m] = self.replacement_coeff * self.o_tpc_cm[m,:m,:m]  
            
                # calculate the size of the product demand for year m
                demand_pr = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
                
                # if the demand is smaller than the potential for component replacement,
                # the oldest products will not receive a new component
                products_for_replacements =  np.einsum('pc -> p', 
                                                       self.replacement_tpc_cm[m,:m,:m])
                p = 0
                while (products_for_replacements.sum() > demand_pr) and (p < m):
                    if products_for_replacements[p+1:].sum() > demand_pr:
                        products_for_replacements[p] = 0
                    else:
                        products_for_replacements[p] = demand_pr - products_for_replacements[p+1:].sum()
                    p += 1
                   
                # Add products with replaced components to stock with a new component cohort
                self.s_tpc[m,:m,m] = products_for_replacements
                                                                   
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m+1].sum()
           
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_pr[m,m] * self.hz_cm[m,c]
                # failure from products only given by multiplying product hazard function and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_pr[m,m] * (1-self.hz_cm[m,c])
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.hz_pr[m,m])

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = (self.o_tpc_pr[m,m,:m+1] 
                                   + self.o_tpc_cm[m,m,:m+1] 
                                   + self.o_tpc_both[m,m,:m+1])

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 

                
        # Calculating aggregated values     
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.oc_pr = np.einsum('tpc->tp', self.o_tpc - self.replacement_tpc_cm)
        self.oc_cm = np.einsum('tpc->tc', self.o_tpc)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)


        
    def case_6(self, max_age_cm_reuse=None):
        '''
        Products and components have independent hazard functions. 
        Potential failure of the component is not included in the hazard function of the product. 
        Components can be reused, and they can be replaced. 
        If the component fails, the product is can get a replacement according to the replacement_coeff. 
        As a first priority, the replacement component would be a reused one from the same cohort. 
        If the reusable components of the same cohort are not enough for the demand of replacement,
        newer cohorts are used. 
        Once all reused components have been installed in replacements,
        and if there is still a demand for component replacement, new components are used. 
        The stock composition of the product is no longer equal 
        to the stock composition of the component, but the total stock is. 
        '''
        
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_pr is  None: 
            raise Exception('No product lifetime specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.replacement_coeff is None:
            raise Exception('No replacement coefficient specified')
        if self.reuse_coeff is None:
            raise Exception('No reuse coefficient specified')
        
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        self.oc_cm = np.zeros((len(self.t), len(self.t))) # component outflows by cohort
        self.oc_pr = np.zeros((len(self.t), len(self.t))) # product outflows by cohort
        
        # component reuse (in old and new products)
        self.reuse_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        if max_age_cm_reuse is None:
            max_age_cm_reuse = len(self.t)

        # component replacement (in old products)
        self.replacement_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        
        self.replace_reuse_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) 
        
        # construct the hazard functions of product and components
        self.compute_sf_pr() # Computes sf if not present already.
        self.compute_sf_cm() # Computes sf of component if not present already.
        self.compute_hz_pr() # product hazard function
        self.compute_hz_cm() # component hazard function
 
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]

        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for c in range(m):     # for all component cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                    self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_pr[m,:m] * self.hz_cm[m,c]
                    # failure from products only given by multiplying product hazard function and
                    # assuming that the probability of component failure is independent from product failure
                    self.o_tpc_pr[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_pr[m,:m] * (1-self.hz_cm[m,c])
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of product failure is independent from product failure
                    self.o_tpc_cm[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,c] * (1-self.hz_pr[m,:m])
                
                # Total outflows for all cohorts <m
                self.o_tpc[m,:m,:m] = (self.o_tpc_pr[m,:m,:m] 
                                    + self.o_tpc_cm[m,:m,:m] 
                                    + self.o_tpc_both[m,:m,:m])
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m] 
                              
                # potential for component reuse
                oldest_cohort = max(m-max_age_cm_reuse, 0)
                self.reuse_tpc_cm[m,:m,oldest_cohort:m] = (self.reuse_coeff
                                                        * self.o_tpc_pr[m,:m,oldest_cohort:m])    
   
                # need for component replacement (in old products)
                self.replacement_tpc_cm[m,:m,:m] = self.replacement_coeff * self.o_tpc_cm[m,:m,:m]    
                
                # calculate the size of the product demand for year m
                demand_pr = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
                
                # if the demand is smaller than the potential for component replacement,
                # the oldest products will not receive a new component
                products_for_replacements =  np.einsum('pc -> p', 
                                                       self.replacement_tpc_cm[m,:m,:m])
                p = 0
                while (products_for_replacements.sum() > demand_pr) and (p < m):
                    if products_for_replacements[p+1:].sum() > demand_pr:
                        products_for_replacements[p] = 0
                    else:
                        products_for_replacements[p] = demand_pr - products_for_replacements[p+1:].sum()
                    p += 1
                
                # if the demand is smaller than the potential for component reuse,
                # the oldest components will not be reused
                components_for_reuse =  np.einsum('pc -> c', 
                                                 self.reuse_tpc_cm[m,:m,oldest_cohort:m])
                c = oldest_cohort
                while (components_for_reuse.sum() > demand_pr) and (c < m):
                    if components_for_reuse[c+1:].sum() > demand_pr:
                        components_for_reuse[c] = 0
                    else:
                        components_for_reuse[c] = demand_pr - components_for_reuse[c+1:].sum()
                    c += 1                     
                                
                # the oldest components are reused in the oldest products
                # in need for a component replacement
                while (p < m) and (c < m): 
                        self.replace_reuse_tpc[m,p,c] = min(
                                products_for_replacements[p],
                                components_for_reuse[c])
                        products_for_replacements[p] -= self.replace_reuse_tpc[m,p,c]
                        components_for_reuse[c] -= self.replace_reuse_tpc[m,p,c]
                        if components_for_reuse[c]==0:
                            c+=1
                        if products_for_replacements[p]==0:
                            p+=1
                                              
                # Add  remaining products to stock with a new component cohort
                self.replace_reuse_tpc[m,p:m,m] = products_for_replacements[p:]
                # Add  remaining components to stock with a new product cohort
                self.replace_reuse_tpc[m,m,c:m] = components_for_reuse[c:]
                # add replaced/reused components to stock:
                self.s_tpc[m,:m+1,:m+1] += self.replace_reuse_tpc[m,:m+1,:m+1]            
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m+1].sum()
     
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_pr[m,m] * self.hz_cm[m,c]
                # failure from products only given by multiplying product hazard function and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_pr[m,m] * (1-self.hz_cm[m,c])
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.hz_pr[m,m])

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = (self.o_tpc_pr[m,m,:m+1]
                                    + self.o_tpc_cm[m,m,:m+1]
                                    + self.o_tpc_both[m,m,:m+1])

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 
            
            # to get outflows by cohorts, we need to consider reused products and components
            self.oc_pr[m,:m] = (np.einsum('pc->p', self.o_tpc[m,:m,:m+1])
                - np.einsum('pc->p', self.replace_reuse_tpc[m,:m,:m+1]))
            self.oc_pr[m,m] = self.o_tpc[m,m,:m+1].sum() 
            self.oc_cm[m,:m] = (np.einsum('pc->c',self.o_tpc[m,:m+1,:m])
                - np.einsum('pc->c', self.replace_reuse_tpc[m,:m+1,:m]))
            self.oc_cm[m,m] = self.o_tpc[m,:m+1,m].sum()

        # Calculating aggregated values                    
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)

    def case_7(self):
        '''
        The model does not use lifetimes at all. It would make sense if the inflows of products and components are calculated as a percentage of 
        the stock (Stock and birth rate model in the paper Lauinger et al.). OK for models where the stock is stable, not suited for studying the 
        penetration of new products and technologies. 1 product = 1 component, no replacements allowed

        This model is driven by a death rate. The component outflows and the product outflows are equivalent and 
        the rate encompases failures in both products. 

        This case computes model using death rate
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.d is None: 
            raise Exception('No death rate specified')
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

                
    def case_7b(self):
        '''
        The model does not use lifetimes at all. It would make sense if the inflows of products and components are calculated as a percentage of 
        the stock (Stock and birth rate model in the paper Lauinger et al.). OK for models where the stock is stable, not suited for studying the 
        penetration of new products and technologies. 1 product = 1 component, no replacements allowed

        This model is driven by a birth rate. The component outflows and the product outflows are equivalent and 
        the rate encompases failures in both products. 

        This case computes model using birth rate
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.b is None: 
            raise Exception('No birth rate specified')
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

    def case_8(self):
        '''
        The model does not use lifetimes at all. It would make sense if the inflows of products and components are calculated as a percentage of the stock 
        (Stock and birth rate model in the paper by Lauinger, Billy, et al.). OK for models where the stock is stable, not suited for studying the penetration of new 
        products and technologies. 
        
        In this case, a replacement rate for the component is considered according to the replacement_coeff.

        This case computes the model using death rate.
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.d is None: 
            raise Exception('No death rate specified')
        if self.replacement_coeff is None:
            raise Exception('No replacement rate specified')
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
            self.o_cm[m] = self.o_pr[m] * self.replacement_coeff # Components die with products      
            self.i_pr[m] = self.ds_pr[m] + self.o_pr[m]
            self.i_cm[m] = self.ds_pr[m] + self.o_cm[m]
            # 3) Add new inflow to stock and determine future decay of new age-cohort
            self.sc_pr[m, m] = self.i_pr[m]
        # 4) Determining the values for the component
        self.sc_cm = self.sc_pr
                
    def case_9(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. No component replacement, when it fails, the product fails.  
        There should be some extra outflows of products as well (accidents), which could be modelled by a stock and death rate approach. 
        1 product = one component, Outflow component = outflow product.  
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.d is None:
            raise Exception('No death rate specified')
        if self.reuse_coeff is None: 
            raise Exception('No replacement rate specified')
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        # construct the sf of a product of cohort tc remaining in the stock in year t
        self.compute_sf_cm() # Computes sf if not present already.
        self.compute_hz_cm() # Computes hazard function of components if not present already
        
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]

        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for c in range(m):     # for all component cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                    self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,:m] * self.d
                    
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of component failure is independent from product failure
                    self.o_tpc_cm[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,:m] * (1-self.d)
                    
                    # failure of products described with death rate
                    # assuming that the probability of product failure is independent from product failure
                    self.o_tpc_pr[m,:m,c] = self.s_tpc[m-1,:m,c] * self.d * (1-self.hz_cm[m,c])
                    
                # Total outflows for all cohorts <m 
                self.o_tpc[m,:m,:m] = self.o_tpc_pr[m,:m,:m] + self.o_tpc_cm[m,:m,:m]  + self.o_tpc_both[m,:m,:m]
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                    
                    
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m+1].sum()
            
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,m] * self.d
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.d)
                # failure from products only given by multiplying death rate and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.d * (1-self.hz_cm[m,m])

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = self.o_tpc_pr[m,m,:m+1] + self.o_tpc_cm[m,m,:m+1] + self.o_tpc_both[m,m,:m+1]

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 
        # Calculating aggregated values      
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.oc_pr  = np.einsum('tpc->tp', self.o_tpc)
        self.oc_cm  = np.einsum('tpc->tc', self.o_tpc)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)


    def case_10(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. No component replacement, when it fails, the product fails, 
        but some components from failed products can be reused in new products.  There should be some extra outflows of products as well (accidents), 
        which could be modelled by a stock and death rate approach. Outflow component <= outflow product. This case is probably not the most meaningful.

        In this case some of the components are deemed to be reusable. This makes sense if we consider that some of the outflows are due to product failures. 
        We can here consider then that only a fraction of the components that would qualify can actually be reused. We do this by defining a reuse rate of the
        components in failed products.
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.d is None:
            raise Exception('No death rate specified')
        if self.reuse_coeff is None: 
            raise Exception('No replacement rate specified')
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        # component reuse (in new products)
        self.reuse_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # construct the sf of a product of cohort tc remaining in the stock in year t
        self.compute_sf_cm() # Computes sf if not present already.
        self.compute_hz_cm() # Computes hazard function of components if not present already
        
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]
        new_inflow_pr = 0

        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for c in range(m):     # for all component cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                    self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,:m] * self.d
                    
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of component failure is independent from product failure
                    self.o_tpc_cm[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,:m] * (1-self.d)
                    
                    # failure of products described with death rate
                    # assuming that the probability of product failure is independent from product failure
                    self.o_tpc_pr[m,:m,c] = self.s_tpc[m-1,:m,c] * self.d * (1-self.hz_cm[m,c])
                    
                # Total outflows for all cohorts <m 
                self.o_tpc[m,:m,:m] = self.o_tpc_pr[m,:m,:m] + self.o_tpc_cm[m,:m,:m]  + self.o_tpc_both[m,:m,:m]
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                
                # component reuse (in new products)
                self.reuse_tpc_cm[m,:m,:m] = self.reuse_coeff * self.o_tpc_pr[m,:m,:m]    
                
                # Add reused components to stock in a new product cohort
                new_inflow_pr = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
                if new_inflow_pr >= self.reuse_tpc_cm[m,:m,:m].sum():
                    self.s_tpc[m,m,:m] = np.einsum('pc -> c', self.reuse_tpc_cm[m,:m,:m])
                else:
                    print(m,' too many components to reuse')
                    
                    
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m+1].sum()
            
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,m] * self.d
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.d)
                # failure from products only given by multiplying death rate and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.d * (1-self.hz_cm[m,m])

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = self.o_tpc_pr[m,m,:m+1] + self.o_tpc_cm[m,m,:m+1] + self.o_tpc_both[m,m,:m+1]

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 
        # Calculating aggregated values      
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.oc_pr  = np.einsum('tpc->tp', self.o_tpc)
        self.oc_cm  = np.einsum('tpc->tc', self.o_tpc - self.reuse_tpc_cm)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)
            


    def case_11(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. 
        The component can be replaced, but old components cannot be reused. 
        This model assumes that a product would live infinitely as long as the component keeps being replaced.
        There should be some extra outflows of products as well (accidents), which could be modelled by a stock and death rate approach. Outflow component >= outflow product.  
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.d is None:
            raise Exception('No death rate specified')
        if self.replacement_coeff is None: 
            raise Exception('No replacement rate specified')
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        # component reuse (in new products)
        self.replacement_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # construct the sf of a product of cohort tc remaining in the stock in year t
        self.compute_sf_cm() # Computes sf if not present already.
        self.compute_hz_cm() # Computes hazard function of components if not present already
        
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]
        new_inflow_pr = 0

        # construct the sf of a product of cohort tc remaining in the stock in year t
        self.compute_sf_cm() # Computes sf if not present already.
        self.compute_hz_cm() # Computes component hazard function
        
        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for c in range(m):     # for all component cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                    self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,:m,c] * self.d * self.hz_cm[m,c]
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of product failure is independent from product failure
                    self.o_tpc_cm[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,c] * (1-self.d)
                    # failure from products only given by multiplying product hazard function and
                    # assuming that the probability of component failure is independent from product failure
                    self.o_tpc_pr[m,:m,c] = self.s_tpc[m-1,:m,c] * self.d * (1-self.hz_cm[m,c])
                
                # Total outflows for all cohorts <m
                self.o_tpc[m,:m,:m] = self.o_tpc_pr[m,:m,:m] + self.o_tpc_cm[m,:m,:m] + self.o_tpc_both[m,:m,:m]
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                
                # component replacement (in old products)
                self.replacement_tpc_cm[m,:m,:m] = self.replacement_coeff * self.o_tpc_cm[m,:m,:m]    
                 
                # Add products with replaced components to stock with a new component cohort
                new_inflow_pr = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
                if new_inflow_pr >= self.replacement_tpc_cm[m,:m,:m].sum():
                    self.s_tpc[m,:m,m] = np.einsum('pc -> p', self.replacement_tpc_cm[m,:m,:m])
                                                    
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m+1].sum()
                
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,m,c] * self.d * self.hz_cm[m,c]
                # failure from products only given by multiplying product hazard function and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.d * (1-self.hz_cm[m,c])
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.d)

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = self.o_tpc_pr[m,m,:m+1] + self.o_tpc_cm[m,m,:m+1] + self.o_tpc_both[m,m,:m+1]

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 

                
        # Calculating aggregated values     
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.oc_pr = np.einsum('tpc->tp', self.o_tpc - self.replacement_tpc_cm)
        self.oc_cm = np.einsum('tpc->tc', self.o_tpc)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)
                
                
    def case_12(self):
        '''
        The lifetime of the product is mostly determined by the lifetime of the component. 
        The component can be replaced, and old components can be reused. 
        This model assumes that a product would live infinitely as long as the component keeps being replaced. 
        There should be some extra outflows of products as well (accidents), which could be modelled by a stock and death rate approach. 
        Outflow component <= or >= outflow product depending on assumptions for reuse and replacement.

        We take a similar approach to case 6 here, only the products do not have a lifetime and therefore are always elegible to get a component replacement. 

        We take the share of components that is exiting use due to product crashes and take a share for reuse according to the reuse_coeff.
         
        '''
        if self.s_pr is None:
            raise Exception('No stock specified')
        if self.lt_cm is None:
            raise Exception('No component lifetime specified')
        if self.d is None:
            raise Exception('No death rate specified')
        if self.replacement_coeff is None: 
            raise Exception('No replacement rate specified')
        if self.reuse_coeff is None: 
            raise Exception('No reuse rate specified')
        # stock composition per year, product cohort and component cohort
        self.s_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) #
        # outflows caused by simultaneous product and component failure:
        self.o_tpc_both = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_pr = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # outflows caused by product failure only:
        self.o_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # total outflows:
        self.o_tpc = np.zeros((len(self.t), len(self.t), len(self.t)))
        self.i_pr = np.zeros(len(self.t)) # product inflows
        self.i_cm = np.zeros(len(self.t)) # component inflows
        # stock change of product:
        self.ds_pr = np.concatenate(([self.s_pr[0]], np.diff(self.s_pr)))
        
        self.o_cm = np.zeros(len(self.t)) # component outflows
        self.o_pr = np.zeros(len(self.t)) # product outflows
        self.oc_cm = np.zeros((len(self.t), len(self.t))) # component outflows by cohort
        self.oc_pr = np.zeros((len(self.t), len(self.t))) # product outflows by cohort
        # component reuse (in new products)
        self.replacement_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        # component reuse (in new products)
        self.reuse_tpc_cm = np.zeros((len(self.t), len(self.t), len(self.t))) 
        self.replace_reuse_tpc = np.zeros((len(self.t), len(self.t), len(self.t))) 

        
        # Initializing values
        self.s_tpc[0,0,0] = self.s_pr[0]

        # construct the sf of a product of cohort tc remaining in the stock in year t
        self.compute_sf_cm() # Computes sf if not present already.
        self.compute_hz_cm() # Computes component hazard function
        
        for m in range(len(self.t)):  # for all years m
            if m>0: # the initial stock is assumed to be 0
                for c in range(m):     # for all component cohorts < m
                    # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                    self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,:m,c] * self.d * self.hz_cm[m,c]
                    
                    # failure from components only given by multiplying component hazard function and
                    # assuming that the probability of product failure is independent from product failure
                    self.o_tpc_cm[m,:m,c] = self.s_tpc[m-1,:m,c] * self.hz_cm[m,c] * (1-self.d)
                    
                    # failure from products only given by multiplying product hazard function and
                    # assuming that the probability of component failure is independent from product failure
                    self.o_tpc_pr[m,:m,c] = self.s_tpc[m-1,:m,c] * self.d * (1-self.hz_cm[m,c])
                
                # Total outflows for all cohorts <m
                self.o_tpc[m,:m,:m] = self.o_tpc_pr[m,:m,:m] + self.o_tpc_cm[m,:m,:m] + self.o_tpc_both[m,:m,:m]
                                
                # subtract outflows of cohorts <m from the previous stock 
                self.s_tpc[m,:m,:m] = self.s_tpc[m-1,:m,:m] - self.o_tpc[m,:m,:m]
                
                # component replacement (in old products)
                self.replacement_tpc_cm[m,:m,:m] = self.replacement_coeff * self.o_tpc_cm[m,:m,:m]   
                
                # component reuse (in new products)
                self.reuse_tpc_cm[m,:m,:m] = self.reuse_coeff * self.o_tpc_pr[m,:m,:m]     
                 
                # calculate the size of the product demand for year m
                demand_pr = self.s_pr[m] - self.s_tpc[m,:m,:m].sum()
                
                # if the demand is smaller than the potential for component replacement,
                # the oldest products will not receive a new component
                products_for_replacements =  np.einsum('pc -> p', 
                                                       self.replacement_tpc_cm[m,:m,:m])
                
                p = 0
                while(products_for_replacements.sum() > demand_pr) and (p < m):
                    if products_for_replacements[p+1:].sum() > demand_pr:
                        products_for_replacements[p] = 0
                    else:
                        products_for_replacements[p] = demand_pr - products_for_replacements[p+1:].sum()
                    p += 1
                
                # if the demand is smaller than the potential for component reuse,
                # the oldest components will not be reused
                components_for_reuse =  np.einsum('pc -> c', 
                                                 self.reuse_tpc_cm[m,:m,:m])
                
                while (components_for_reuse.sum() > demand_pr) and ( c< m):
                    if components_for_reuse[c+1:].sum() > demand_pr:
                        components_for_reuse[c] = 0
                    else:
                        components_for_reuse[c] = demand_pr - components_for_reuse[c+1:].sum()
                    c += 1       
                    
                # the oldest components are reused in the oldest products
                # in need for a component replacement
                while (p < m) and (c < m): 
                        self.replace_reuse_tpc[m,p,c] = min(
                                products_for_replacements[p],
                                components_for_reuse[c])
                        products_for_replacements[p] -= self.replace_reuse_tpc[m,p,c]
                        components_for_reuse[c] -= self.replace_reuse_tpc[m,p,c]
                        if components_for_reuse[c]==0:
                            c+=1
                        if products_for_replacements[p]==0:
                            p+=1
                            
                             
                # Add  remaining products to stock with a new component cohort
                self.replace_reuse_tpc[m,p:m,m] = products_for_replacements[p:]
                # Add  remaining components to stock with a new product cohort
                self.replace_reuse_tpc[m,m,c:m] = components_for_reuse[c:]
                # add replaced/reused components to stock:
                self.s_tpc[m,:m+1,:m+1] += self.replace_reuse_tpc[m,:m+1,:m+1]            
                # Add new product cohort with new components to stock 
                self.s_tpc[m,m,m] = self.s_pr[m] - self.s_tpc[m,:m+1,:m+1].sum()
                
            # Calculate outflows in the first year for product cohort m
            for c in range(m):     # for all component cohorts < m
                # simultaneaous failure given by multiplying both hazard functions for all cohorts <m
                self.o_tpc_both[m,:m,c] = self.s_tpc[m-1,m,c] * self.d * self.hz_cm[m,c]
                # failure from products only given by multiplying product hazard function and
                # assuming that the probability of component failure is independent from product failure
                self.o_tpc_pr[m,m,c] = self.s_tpc[m-1,m,c] * self.d * (1-self.hz_cm[m,c])
                # failure from components only given by multiplying component hazard function and
                # assuming that the probability of product failure is independent from product failure
                self.o_tpc_cm[m,m,c] = self.s_tpc[m-1,m,c] * self.hz_cm[m,c] * (1-self.d)

            # Total outflows for cohort m
            self.o_tpc[m,m,:m+1] = self.o_tpc_pr[m,m,:m+1] + self.o_tpc_cm[m,m,:m+1] + self.o_tpc_both[m,m,:m+1]

            # # Calculate new product inflow, accounting for outflows in the first year
            self.i_pr[m] = self.s_tpc[m,m,:m+1].sum()  + self.o_tpc[m,m,:m+1].sum()

            # # Calculate new component inflow, accounting for outflows in the first year                    
            self.i_cm[m] = self.s_tpc[m,:m+1,m].sum()  + self.o_tpc[m,:m+1,m].sum() 
            
            self.oc_pr[m,:m] = np.einsum('pc->p', self.o_tpc[m,:m,:m+1]) - np.einsum('pc->p', self.replace_reuse_tpc[m,:m,:m+1])
            self.oc_pr[m,m] = self.o_tpc[m,m,:m+1].sum() 
            self.oc_cm[m,:m] = np.einsum('pc->c',self.o_tpc[m,:m+1,:m]) - np.einsum('pc->c', self.replace_reuse_tpc[m,:m+1,:m])
            self.oc_cm[m,m] = self.o_tpc[m,:m+1,m].sum()

                
        # Calculating aggregated values     
        self.o_pr = self.i_pr - self.ds_pr 
        self.o_cm = self.i_cm - self.ds_pr 
        self.oc_pr = np.einsum('tpc->tp', self.o_tpc - self.replacement_tpc_cm)
        self.oc_cm = np.einsum('tpc->tc', self.o_tpc)
        self.sc_pr  = np.einsum('tpc->tp', self.s_tpc)
        self.sc_cm  = np.einsum('tpc->tc', self.s_tpc)

