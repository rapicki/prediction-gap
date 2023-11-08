import shap
import numpy as np
import xgboost as xgb
import pandas as pd


class ShapWrapper:
   
   @staticmethod
   def _get_ranking_by_name(index_ranking: np.array, names: list[str]) -> list:
        names = np.array(names)
        ranking = names[index_ranking].tolist()
        return ranking


   def get_shap_ranking(self, xgb_tree: xgb.Booster, X: pd.DataFrame, return_str = True) -> list:
        explainer = shap.TreeExplainer(xgb_tree)
        shap_values = explainer.shap_values(X)
        ranking = np.argsort(shap_values)
        if return_str:
            ranking = self._get_ranking_by_name(ranking, list(X.columns.values))
        else:
            ranking = ranking.tolist() 
        
        return ranking
