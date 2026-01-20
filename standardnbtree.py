# Standard NBTree

import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

random.seed(18)

dataMap = {}
attrNames = []
targetName = ""
minLeafStop = 5

def calcEntropy(indices):
    counts = {}
    for idx in indices:
        lbl = dataMap[idx][targetName]
        counts[lbl] = counts.get(lbl, 0) + 1
    total = float(len(indices))
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent

def splitByAttribute(indices, attr):
    parts = {}
    for idx in indices:
        val = dataMap[idx][attr]
        if val not in parts:
            parts[val] = []
        parts[val].append(idx)
    return parts

def calcInfoAfterSplit(indices, attr):
    parts = splitByAttribute(indices, attr)
    total = float(len(indices))
    info = 0.0
    for subset in parts.values():
        info += (len(subset) / total) * calcEntropy(subset)
    return info, parts

def calcSplitInfo(indices, parts):
    total = float(len(indices))
    si = 0.0
    for subset in parts.values():
        r = len(subset) / total
        if r > 0:
            si -= r * math.log(r, 2)
    return si

def gainRatio(indices, availableAttrs):
    baseEnt = calcEntropy(indices)
    bestAttr = None
    bestParts = None
    bestScore = -1.0
    for a in availableAttrs:
        infoA, parts = calcInfoAfterSplit(indices, a)
        gain = baseEnt - infoA
        si = calcSplitInfo(indices, parts)
        score = gain / si if si > 0 else gain
        if score > bestScore:
            bestScore = score
            bestAttr = a
            bestParts = parts
    return bestAttr, bestParts

def getMajorityLabel(indices):
    counts = {}
    for idx in indices:
        lab = dataMap[idx][targetName]
        counts[lab] = counts.get(lab, 0) + 1
    bestLabel = None
    bestCount = -1
    for lab, c in counts.items():
        if c > bestCount:
            bestCount = c
            bestLabel = lab
    return bestLabel

def decisionTree(indices, availableAttrs):
    labels = [dataMap[idx][targetName] for idx in indices]
    if len(indices) < minLeafStop:
        return {"type": "leaf", "label": getMajorityLabel(indices)}
    if len(set(labels)) == 1:
        return {"type": "leaf", "label": labels[0]}
    if not availableAttrs:
        return {"type": "leaf", "label": getMajorityLabel(indices)}
    bestAttr, parts = gainRatio(indices, availableAttrs)
    if bestAttr is None:
        return {"type": "leaf", "label": getMajorityLabel(indices)}
    node = {"type": "node", "attr": bestAttr, "branches": {}}
    for val, subset in parts.items():
        if not subset:
            node["branches"][val] = {"type": "leaf", "label": getMajorityLabel(indices)}
        else:
            nextAttrs = [a for a in availableAttrs if a != bestAttr]
            node["branches"][val] = decisionTree(subset, nextAttrs)
    return node

def findLeafNode(node, instance):
    if node["type"] == "leaf":
        return node
    val = instance.get(node["attr"])
    child = node["branches"].get(val)
    if child is None:
        keys = sorted(list(node["branches"].keys()))
        if not keys:
            return {"type": "leaf", "label": getMajorityLabel([k for k in dataMap],)}
        child = node["branches"][keys[0]]
    return findLeafNode(child, instance)

def trainNaiveBayesDiscrete(indices, featureList, laplace):
    classes = []
    for idx in indices:
        lab = dataMap[idx][targetName]
        if lab not in classes:
            classes.append(lab)
    classes.sort()
    classIndex = {c: i for i, c in enumerate(classes)}
    nClasses = len(classes)
    n = len(indices)
    counts = [0] * nClasses
    for idx in indices:
        c = dataMap[idx][targetName]
        counts[classIndex[c]] += 1
    priors = [ (counts[i] + laplace) / (n + laplace * nClasses) for i in range(nClasses) ]
    condProb = {}
    for attr in featureList:
        valuesList = []
        for idx in indices:
            v = dataMap[idx][attr]
            if v not in valuesList:
                valuesList.append(v)
        valuesList.sort()
        nValues = len(valuesList)
        mat = [ [0] * nValues for _ in range(nClasses) ]
        for idx in indices:
            v = dataMap[idx][attr]
            c = dataMap[idx][targetName]
            iClass = classIndex[c]
            iVal = valuesList.index(v)
            mat[iClass][iVal] += 1
        probMat = []
        for iClass in range(nClasses):
            denom = counts[iClass] + laplace * nValues
            row = [ (mat[iClass][j] + laplace) / denom for j in range(nValues) ]
            probMat.append(row)
        condProb[attr] = (valuesList, probMat)
    return classes, priors, condProb

def predictNaiveBayesInstance(instance, classes, priors, condProb, featureList, laplace):
    nClasses = len(classes)
    logScores = [0.0] * nClasses
    for i in range(nClasses):
        p = priors[i]
        if p <= 0:
            p = 1e-12
        logScores[i] = math.log(p)
    for attr in featureList:
        if attr not in condProb:
            continue
        valuesList, probMat = condProb[attr]
        for iClass in range(nClasses):
            v = instance.get(attr)
            if v in valuesList:
                j = valuesList.index(v)
                probVal = probMat[iClass][j]
            else:
                probVal = 1e-12
            logScores[iClass] += math.log(probVal)
    maxLog = max(logScores)
    exps = [math.exp(s - maxLog) for s in logScores]
    ssum = sum(exps)
    probs = [e / ssum for e in exps]
    return probs

def buildNBTreeStandard(trainIndices, availableAttrs, laplace, minSamplesForNB):
    tree = decisionTree(trainIndices, availableAttrs)
    leafToIndices = {}
    for idx in trainIndices:
        inst = dataMap[idx]
        leaf = findLeafNode(tree, inst)
        lid = id(leaf)
        if lid not in leafToIndices:
            leafToIndices[lid] = []
        leafToIndices[lid].append(idx)
    leafModels = {}
    leafSizes = []
    for lid, inds in leafToIndices.items():
        nLeaf = len(inds)
        leafSizes.append(nLeaf)
        nbModel = None
        if nLeaf >= minSamplesForNB:
            featureList = list(availableAttrs)
            classes, priors, condProb = trainNaiveBayesDiscrete(inds, featureList, laplace)
            nbModel = {"classes": classes, "priors": priors, "condProb": condProb, "features": featureList}
        counts = {}
        for idx in inds:
            c = dataMap[idx][targetName]
            counts[c] = counts.get(c, 0) + 1
        total = float(len(inds))
        pLeaf = {}
        for c, cnt in counts.items():
            pLeaf[c] = cnt / total if total > 0 else 0.0
        leafModels[lid] = {"n": nLeaf, "nbModel": nbModel, "pLeaf": pLeaf}
    return tree, leafModels, leafSizes

def predictWithNBTreeStandard(treeRoot, leafModels, instances, laplace):
    preds = []
    probsAll = []
    for inst in instances:
        leaf = findLeafNode(treeRoot, inst)
        lid = id(leaf)
        model = leafModels.get(lid)
        if model is None or model["nbModel"] is None:
            counts = {}
            for k in dataMap:
                c = dataMap[k][targetName]
                counts[c] = counts.get(c, 0) + 1
            maj = max(counts.items(), key=lambda x: x[1])[0] if counts else None
            preds.append(maj)
            probsAll.append({maj: 1.0} if maj is not None else {})
            continue
        nbModel = model["nbModel"]
        classes = nbModel["classes"]
        priors = nbModel["priors"]
        condProb = nbModel["condProb"]
        featureList = nbModel["features"]
        nbProbsList = predictNaiveBayesInstance(inst, classes, priors, condProb, featureList, laplace)
        probMap = {}
        for i, cl in enumerate(classes):
            probMap[cl] = nbProbsList[i]
        best = max(probMap.items(), key=lambda x: x[1])[0] if probMap else None
        preds.append(best)
        probsAll.append(probMap)
    return preds, probsAll

def dataToDict(table, targetColumn=None):
    if targetColumn is None:
        targetColumn = table.columns[-1]
    attrsLocal = [c for c in table.columns if c != targetColumn]
    dataDictLocal = {}
    for idx in range(len(table)):
        row = table.iloc[idx]
        inst = {}
        for a in attrsLocal:
            inst[str(a).strip()] = str(row[a]).strip()
        inst[str(targetColumn).strip()] = str(row[targetColumn]).strip()
        dataDictLocal[idx+1] = inst
    return dataDictLocal, [str(a).strip() for a in attrsLocal], str(targetColumn).strip()

def getMajorityFromTrain():
    counts = {}
    for k in dataMap:
        c = dataMap[k][targetName]
        counts[c] = counts.get(c, 0) + 1
    mostFreq = max(counts.items(), key=lambda x: x[1])[0] if counts else None
    return mostFreq

def main():
    global dataMap, attrNames, targetName, minLeafStop
    laplace = 1.0
    minSamplesForNB = 2
    minLeafStop = 5
    FILENAME = "iris.csv"

    print(f"Dataset: {FILENAME}")
    IN = "Datasets/" + FILENAME
    df = pd.read_csv(IN)

    allDataDict, attrNames, targetName = dataToDict(df)

    allIndices = list(allDataDict.keys())

    kf = KFold(n_splits=10, shuffle=True, random_state=18)

    allAcc, allPrec, allRec, allF1 = [], [], [], []
    allClasses = sorted(list(set([allDataDict[k][targetName] for k in allIndices])))
    aggCM = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(allIndices), 1):
        trainIDs = [allIndices[i] for i in train_idx]
        testIDs  = [allIndices[i] for i in test_idx]

        dataMap = {k: allDataDict[k] for k in trainIDs}
        testInstances = [allDataDict[k] for k in testIDs]
        testActuals = [allDataDict[k][targetName] for k in testIDs]

        treeStd, leafModelsStd, _ = buildNBTreeStandard(trainIDs, attrNames, laplace, minSamplesForNB)

        predsStd, _ = predictWithNBTreeStandard(treeStd, leafModelsStd, testInstances, laplace)
        predsStdFixed = [(p if p is not None else getMajorityFromTrain()) for p in predsStd]

        allAcc.append(accuracy_score(testActuals, predsStdFixed))
        allPrec.append(precision_score(testActuals, predsStdFixed, average='weighted', zero_division=0))
        allRec.append(recall_score(testActuals, predsStdFixed, average='weighted', zero_division=0))
        allF1.append(f1_score(testActuals, predsStdFixed, average='weighted', zero_division=0))
        cm = confusion_matrix(testActuals, predsStdFixed, labels=allClasses)
        if aggCM is None:
            aggCM = cm
        else:
            aggCM += cm
        print(f"Fold {fold} completed: Acc={allAcc[-1]:.4f}, F1={allF1[-1]:.4f}")

    print("\n10-Fold Cross-Validation Results:")
    print(f"Accuracy: {np.mean(allAcc):.4f}")
    print(f"Precision: {np.mean(allPrec):.4f}")
    print(f"Recall: {np.mean(allRec):.4f}")
    print(f"F1 Score: {np.mean(allF1):.4f}")

    print(aggCM)
if __name__ == "__main__":
    main()