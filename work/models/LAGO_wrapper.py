from work.models.model_wrapper import *
import epftoolbox

class LAGOWrapper(ModelWrapper):
    """
    Only for using as an interface to load forecasts
    """
    def __init__(self, prefix, dataset_name, label):
        ModelWrapper.__init__(self, prefix, dataset_name, label)

    def _remove_country(self, path):
        s = ""
        for ss in path.split("EPF_")[-1].split("_")[1:]: s += ss + "_"
        s = s[:-1]

        res = ""
        for ss in path.split("EPF_")[:-1]: res += ss + "EPF_"
        res = res[:-4]
        
        res += s
        return res
        
    def test_prediction_path(self):
        return self._remove_country(mu.test_prediction_path(self.prefix, self.dataset_name))

    def val_prediction_path(self):
        return self._remove_country(mu.val_prediction_path(self.prefix, self.dataset_name))

    def test_recalibrated_prediction_path(self):
        return self._remove_country(mu.test_recalibrated_prediction_path(
            self.prefix, self.dataset_name))

    def string(self):
        return self.prefix
