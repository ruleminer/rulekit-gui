import pandas as pd
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
