import numpy as np
import scipy
import pandas as pd
from ProcessingData.ExternalCode import EngbertMicrosaccadeToolbox

class EventDetection():
    @staticmethod
    def find_micsacc():
        micsacc = microsac_detection()
