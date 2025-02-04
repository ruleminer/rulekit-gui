import pandas as pd
import streamlit as st
from rulekit.params import Measures


MEASURE_SELECTION = pd.DataFrame({
    "Metric": ["Accuracy", "Binary Entropy", "C1", "C2", "C Foil",
               "CN2 Significnce", "Correlation", "F Bayesian Confirmation",
               "F Measure",  "FullCoverage", "GeoRSS", "GMeasure",
               "InformationGain", "JMeasure", "Kappa", "Klosgen", "Laplace",
               "Lift", "LogicalSufficiency", "MEstimate", "MutualSupport",
               "Novelty", "OddsRatio", "OneWaySupport", "PawlakDependencyFactor",
               "Q2", "Precision", "RelativeRisk", "Ripper", "RuleInterest",
               "RSS", "SBayesian", "Sensitivity", "Specificity", "TwoWaySupport",
               "WeightedLaplace", "WeightedRelativeAccuracy", "YAILS", "LogRank"],
    "Desc": [Measures.Accuracy, Measures.BinaryEntropy, Measures.C1, Measures.C2, Measures.CFoil,
             Measures.CN2Significnce, Measures.Correlation, Measures.FBayesianConfirmation,
             Measures.FMeasure, Measures.FullCoverage, Measures.GeoRSS, Measures.GMeasure,
             Measures.InformationGain, Measures.JMeasure, Measures.Kappa, Measures.Klosgen, Measures.Laplace,
             Measures.Lift, Measures.LogicalSufficiency, Measures.MEstimate, Measures.MutualSupport,
             Measures.Novelty, Measures.OddsRatio, Measures.OneWaySupport, Measures.PawlakDependencyFactor,
             Measures.Q2, Measures.Precision, Measures.RelativeRisk, Measures.Ripper, Measures.RuleInterest,
             Measures.RSS, Measures.SBayesian, Measures.Sensitivity, Measures.Specificity, Measures.TwoWaySupport,
             Measures.WeightedLaplace, Measures.WeightedRelativeAccuracy, Measures.YAILS, Measures.LogRank]
})


def get_measures_selection_dict():
    met_ind_cluss = st.selectbox(
        "Induction measure", MEASURE_SELECTION, index=6)
    ind_cluss = MEASURE_SELECTION.Desc[MEASURE_SELECTION["Metric"]
                                       == met_ind_cluss].values[0]

    met_prun_cluss = st.selectbox(
        "Pruning measure", MEASURE_SELECTION, index=6)
    prun_cluss = MEASURE_SELECTION.Desc[MEASURE_SELECTION["Metric"]
                                        == met_prun_cluss].values[0]

    met_vot_cluss = st.selectbox("Voting measure", MEASURE_SELECTION, index=6)
    vot_cluss = MEASURE_SELECTION.Desc[MEASURE_SELECTION["Metric"]
                                       == met_vot_cluss].values[0]

    dictionary = {"induction_measure": ind_cluss,
                  "pruning_measure": prun_cluss,
                  "voting_measure": vot_cluss, }

    return dictionary
