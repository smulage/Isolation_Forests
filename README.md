# Isolation_Forests - Feature importances

Isolation Forest is a tree based solution for Anomaly Detection. Most models used in outlier detection (eg: clustering) train for similarities and allow the observations with high training errors to be deemed outliers. Isolation Forest however trains specifically to detect outliers as being observations that lie at the short paths of its trees. This model is lite (~100 independently trained trees), but prone to false positives. 

Model Desc: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

False Positives & proposed solution: https://arxiv.org/pdf/1811.02141.pdf

For explainability of results, an outlier's feature importance can be gleaned from a random forest generated off the same dataset with only the outlier set as the target variable. We believe that one can also leverage the Isolation Forest itself by identifying the subset of short paths that contain the outlier and computing feature frequencies

In this analysis, we are vetting the second method. A quick walkthrough of the process:
1) we ran the Isolation Forest for a combination of (tree size, random states) to identify where the top outlier's feature frequencies stabilizes
    tree sizes: 50, 100, 500, 1000
    random states: 20 randomly generated seeds
2) an outlier's feature importances was generated based on the frequency of the feature's sighting along the short paths where the observation lies. The higher the frequency, the more important the feature is deemed.

**Highlights/Conclusions:**
1) sklearn's export_text function generates a text layout of each tree. The recurse_tree function in this repository extracts all the paths.
2) the 'results matrix' image confirms that the forest reaches optimal fit for outliers at around 100 trees, but requires close to 500 trees for feature frequencies. In other words, larger forests lead to a significant noise reduction in generating feature frequencies.
3) on a standard machine, AEN's spyder was able to generate 20 forests of a 1000 trees each in 1 minute


Further enhancements that could be applied:
1) feature frequencies was a simple tabulation of count(feature) along all short paths. Weights could be applied to features based on path length. This might help reduce noise earlier leading to smaller forests.
2) feature frequencies was analyzed for only the model's top outlier. We can go further down the rankings to confirm that the concentration persists.
3) the models' precision was ~80%. Feature engineering could help improve that metric and/or feature frequencies.
4) we employed sklearn's isolation forest. To minimize false positives, one can explore the extended isolation forest available here: https://github.com/sahandha/eif
