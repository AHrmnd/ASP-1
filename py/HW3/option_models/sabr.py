# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    time_step = 0.01
    samples = 10000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None): 
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        b_vol = self.bsm_model.impvol(price = self.price(strike, spot, texp=texp ,sigma=sigma),strike = strike, spot = spot, texp = texp)
        return b_vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1):  
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''   
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac  
        n = int(texp/self.time_step)+1
        
        np.random.seed(123465)
        znorm_m = np.random.normal(size = (n,self.samples))
        np.random.seed(123456)
        ynorm_m = np.random.normal(size = (n,self.samples))
        wnorm_m = self.rho*znorm_m + ynorm_m * np.sqrt(1-self.rho*self.rho)
        
        sigma_t,prices = np.array([self.sigma]*self.samples),np.array([forward]*self.samples)
        for i in range(n):
            sigma_t = sigma_t*np.exp(-0.5 * self.time_step * self.vov * self.vov + self.vov * np.sqrt(self.time_step) * znorm_m[i,:])
            prices = prices*np.exp(-0.5 * self.time_step * sigma_t * sigma_t + sigma_t * np.sqrt(self.time_step) * wnorm_m[i,:])
        option_prices = np.zeros_like(strike)
        for j in range(strike.size):
            option_prices[j] = np.mean(np.fmax(prices - strike[j], 0),axis = 0) 
        
        return option_prices

'''
MC model class for Beta=0
'''

class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    time_step = 0.01
    samples = 10000
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        n_vol = self.normal_model.impvol(price = self.price(strike, spot, texp=texp ,sigma=sigma),strike = strike, spot = spot, texp = texp)
        return n_vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac  
        n = int(texp/self.time_step)+1
        
        np.random.seed(123465)
        znorm_m = np.random.normal(size = (n,self.samples))
        np.random.seed(123445)
        ynorm_m = np.random.normal(size = (n,self.samples))
        wnorm_m = self.rho*znorm_m + ynorm_m * np.sqrt(1-self.rho*self.rho)
        
        #die dai st sigmat
        sigma_t,prices = np.array([self.sigma]*self.samples),np.array([forward]*self.samples)
        for i in range(n):
            sigma_t = sigma_t*np.exp(-0.5 * self.time_step * self.vov * self.vov + self.vov * np.sqrt(self.time_step) * znorm_m[i,:])
            prices = prices + sigma_t * np.sqrt(self.time_step) * wnorm_m[i,:]
        option_prices = np.zeros_like(strike)
        for j in range(strike.size):
            option_prices[j] = np.mean(np.fmax(prices - strike[j], 0),axis = 0)
        return option_prices

import scipy.integrate as spint
'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    time_step = 0.1
    samples = 10000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        b_vol = self.bsm_model.impvol(price = self.price(strike, spot, texp=texp ,sigma=sigma),strike = strike, spot = spot, texp = texp)
        return b_vol
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac  
        n = int(texp/self.time_step)+1
        
        np.random.seed(123465)
        znorm_m = np.random.normal(size = (n,self.samples))  
        np.random.seed(123456)
        xnorm_m = np.random.normal(size = (n,self.samples))  
        sigma_path = np.zeros_like(znorm_m)+self.sigma
 
        for i in range(n-1):
            sigma_path[i+1,:] = sigma_path[i,:]*np.exp(-0.5 * self.time_step * self.vov * self.vov + self.vov * np.sqrt(self.time_step) * znorm_m[i,:])
        
        #integrated
        vt = spint.simps(sigma_path**2, dx=self.time_step,axis=0)
        it = vt/(texp*self.sigma*self.sigma)

        sigma_t = np.mean(sigma_path[-1,:],axis=0)
        vt_t = np.mean(vt,axis=0)
        it_t = np.mean(it,axis=0)

        #Method1: stock price on T
        #np.random.seed(1234567)
        #x1 = np.random.normal(size = n) 
        #prices = forward * np.exp(self.rho/self.vov*(sigma_t-self.sigma)-0.5*vt_t+x1*np.sqrt((1-self.rho*self.rho)*vt_t))
        #option_prices = np.zeros_like(strike)
        #for j in range(strike.size):
        #    option_prices[j] = np.mean(np.fmax(prices - strike[j], 0),axis = 0)
        
        #Method2: put into bs
        self.bsm_model.sigma = self.sigma*np.sqrt((1-self.rho**2)*it_t)
        print(self.bsm_model.__dict__)
        option_prices = self.bsm_model.price(strike, forward * np.exp(self.rho/self.vov*(sigma_t-self.sigma)
                                   -0.5*(self.rho**2)*(self.sigma**2)*texp*it_t), texp)                
         
        return option_prices

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    time_step = 0.1
    samples = 10000
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        ''' 
        n_vol = self.normal_model.impvol(price = self.price(strike, spot, texp=texp ,sigma=sigma),strike = strike, spot = spot, texp = texp)
        return n_vol
        
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac  
        n = int(texp/self.time_step)+1
        
        np.random.seed(123465)
        znorm_m = np.random.normal(size = (n,self.samples))  
        np.random.seed(123456)
        xnorm_m = np.random.normal(size = (n,self.samples))  
        sigma_path = np.zeros_like(znorm_m)+self.sigma
 
        for i in range(n-1):
            sigma_path[i+1] = sigma_path[i]*np.exp(-0.5 * self.time_step * self.vov * self.vov + self.vov * np.sqrt(self.time_step) * znorm_m[i,:])
 
        #integrated
        vt = spint.simps(sigma_path**2, dx=self.time_step,axis=0)
        it = vt/(texp*self.sigma*self.sigma)

        sigma_t = np.mean(sigma_path[-1,:],axis=0)
        vt_t = np.mean(vt,axis=0)
        it_t = np.mean(it,axis=0)
        
        #put into normal
        self.normal_model.sigma = self.sigma*np.sqrt((1-self.rho**2)*it_t)
        option_prices = self.normal_model.price(strike, forward+self.rho/self.vov*(sigma_t-self.sigma), texp)

        return option_prices
