import numpy as np
import matplotlib.pyplot as plt

from particle import literals as lp  # --> MeV units
import hepunits as u  # converting to GeV
import dill as pickle

from Mu3e import fastmc as fm
from Mu3e import mudecays
from Mu3e import plot_tools as pt

from Mu3e import models

baseline_model = models.DS(
    maprime=30, mphi=90, Femu=1e-14, alphaD=1 / 137, epsilon=1e-4
)

NEVENTS = int(1e4)

mu1e2nu = mudecays.Process(channel="mu1e2nu", n_events=NEVENTS)
mu1e2nu.initialize_amplitude()
mu1e2nu.generate()
mu1e2nu.evaluate_amplitude()
mu1e2nu.place_it_in_Mu3e()
pickle.dump(mu1e2nu, open("May_13_mu1e2nu.pkl", "wb"))

mu3e2nu = mudecays.Process(channel="mu3e2nu", n_events=NEVENTS)
mu3e2nu.initialize_amplitude()
mu3e2nu.generate()
mu3e2nu.evaluate_amplitude()
mu3e2nu.place_it_in_Mu3e()
pickle.dump(mu3e2nu, open("May_13_mu3e2nu.pkl", "wb"))

mu5e2nu = mudecays.Process(channel="mu5e2nu", n_events=NEVENTS)
mu5e2nu.initialize_amplitude()
mu5e2nu.generate()
mu5e2nu.evaluate_amplitude()
mu5e2nu.place_it_in_Mu3e()
pickle.dump(mu5e2nu, open("May_13_mu5e2nu.pkl", "wb"))

# New physics
mu5e = mudecays.Process(channel="mu5e", model=baseline_model, n_events=NEVENTS)
mu5e.initialize_amplitude()
mu5e.generate()
mu5e.evaluate_amplitude()
mu5e.place_it_in_Mu3e()
pickle.dump(mu5e, open("May_13_mu5eu.pkl", "wb"))
