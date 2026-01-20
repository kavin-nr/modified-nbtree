# NBTree with Modifications

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
minLeafStop = 30

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

def collectLeafPathFeatures(node, pathFeatures):
    if node["type"] == "leaf":
        return {id(node): set(pathFeatures)}
    res = {}
    attr = node["attr"]
    for val, child in node["branches"].items():
        newPath = list(pathFeatures)
        newPath.append(attr)
        res.update(collectLeafPathFeatures(child, newPath))
    return res

def mutualInformationAttr(indices, attr):
    N = float(len(indices))
    if N == 0:
        return N
    jointCounts = {}
    valueCounts = {}
    classCounts = {}
    for idx in indices:
        v = dataMap[idx][attr]
        c = dataMap[idx][targetName]
        valueCounts[v] = valueCounts.get(v, 0) + 1
        classCounts[c] = classCounts.get(c, 0) + 1
        jointCounts[(v,c)] = jointCounts.get((v,c), 0) + 1
    mi = 0.0
    for (v,c), joint in jointCounts.items():
        p_vc = joint / N
        p_v = valueCounts[v] / N
        p_c = classCounts[c] / N
        if p_v > 0 and p_c > 0:
            ratio = p_vc / (p_v * p_c)
            if ratio > 0:
                mi += p_vc * math.log(ratio, 2)
    return mi

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
                denom = 1.0
                probVal = laplace / (denom + laplace * max(1, len(valuesList)))

            logScores[iClass] += math.log(probVal)
    maxLog = max(logScores)
    exps = [math.exp(s - maxLog) for s in logScores]
    ssum = sum(exps)
    probs = [e / ssum for e in exps]
    return probs

def buildLeafModels(treeRoot, trainIndices, selK, laplace, smoothC, minSamplesForNB):
    leafPathMap = collectLeafPathFeatures(treeRoot, [])
    leafToIndices = {}
    for idx in trainIndices:
        inst = dataMap[idx]
        leafNode = findLeafNode(treeRoot, inst)
        lid = id(leafNode)
        if lid not in leafToIndices:
            leafToIndices[lid] = []
        leafToIndices[lid].append(idx)
    leafModels = {}
    leafSizes = []
    for lid, inds in leafToIndices.items():
        nLeaf = len(inds)
        leafSizes.append(nLeaf)
        pathFeatures = list(leafPathMap.get(lid, set()))
        extraFeatures = []
        if selK > 0 and nLeaf >= max(3, minSamplesForNB):
            miList = []
            for a in attrNames:
                if a in pathFeatures:
                    continue
                mi = mutualInformationAttr(inds, a)
                miList.append((mi, a))
            miList.sort(reverse=True)
            topk = []
            for t in miList[:selK]:
                topk.append(t[1])
            extraFeatures = topk
        localFeatures = list(pathFeatures) + extraFeatures
        nbModel = None
        classes = []
        priors = []
        condProb = {}
        if nLeaf >= minSamplesForNB and len(localFeatures) > 0:
            classes, priors, condProb = trainNaiveBayesDiscrete(inds, localFeatures, laplace)
            nbModel = {"classes": classes, "priors": priors, "condProb": condProb, "features": localFeatures}
        counts = {}
        for idx in inds:
            c = dataMap[idx][targetName]
            counts[c] = counts.get(c, 0) + 1
        total = float(len(inds))
        pLeaf = {}
        for c, cnt in counts.items():
            pLeaf[c] = cnt / total if total > 0 else 0.0
        leafModels[lid] = {"n": nLeaf, "nbModel": nbModel, "pLeaf": pLeaf, "localFeatures": localFeatures}
    return leafModels, leafSizes

def trainGlobalNaiveBayes(trainIndices, laplace):
    featureList = list(attrNames)
    classes, priors, condProb = trainNaiveBayesDiscrete(trainIndices, featureList, laplace)
    return {"classes": classes, "priors": priors, "condProb": condProb, "features": featureList}

def predictGlobalNaiveBayes(nbModel, instance, laplace):
    classes = nbModel["classes"]
    priors = nbModel["priors"]
    condProb = nbModel["condProb"]
    featureList = nbModel["features"]
    probs = predictNaiveBayesInstance(instance, classes, priors, condProb, featureList, laplace)
    best = classes[0]
    bestProb = -1.0
    for i in range(len(classes)):
        if probs[i] > bestProb:
            bestProb = probs[i]
            best = classes[i]
    probMap = {}
    for i in range(len(classes)):
        probMap[classes[i]] = probs[i]
    return best, probMap

def predictWithNBTree(treeRoot, leafModels, instances, laplace, smoothC, globalNB):
    preds = []
    probsAll = []

    for inst in instances:
        leafNode = findLeafNode(treeRoot, inst)
        lid = id(leafNode)
        model = leafModels.get(lid)

        if model is not None and model["nbModel"] is not None:
            nbModel = model["nbModel"]
            localFeatures = nbModel["features"]
            nbProbsList = predictNaiveBayesInstance(inst, nbModel["classes"], nbModel["priors"], nbModel["condProb"], localFeatures, laplace)
            weight = float(model["n"]) / (model["n"] + smoothC)
            finalProbs = {}
            for i, cl in enumerate(nbModel["classes"]):
                pnb = nbProbsList[i]
                pleaf = model["pLeaf"].get(cl, 0.0)
                finalProbs[cl] = weight * pnb + (1.0 - weight) * pleaf
            s = sum(finalProbs.values())
            if s > 0:
                for cl in finalProbs:
                    finalProbs[cl] /= s
            best = max(finalProbs.items(), key=lambda x: x[1])[0]
            preds.append(best)
            probsAll.append(finalProbs)
        else:
            best, probMap = predictGlobalNaiveBayes(globalNB, inst, laplace)
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

def main():
    global dataMap, attrNames, targetName, minLeafStop

    selK = 5
    laplace = 1.0
    smoothC = 5.0
    minSamplesForNB = 10
    minLeafStop = 30
    FILENAME = "iris.csv"
    IN = "Datasets/" + FILENAME
    df = pd.read_csv(IN)
    print(f"Dataset: {FILENAME}")

    allDataDict, attrNames, targetName = dataToDict(df)
    allIndices = list(allDataDict.keys())
    allClasses = sorted(list(set([allDataDict[k][targetName] for k in allIndices])))

    kf = KFold(n_splits=10, shuffle=True, random_state=18)

    allAcc, allPrec, allRec, allF1 = [], [], [], []
    aggCM = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(allIndices), 1):
        trainIDs = [allIndices[i] for i in train_idx]
        testIDs  = [allIndices[i] for i in test_idx]

        dataMap = {k: allDataDict[k] for k in trainIDs}
        testInstances = [allDataDict[k] for k in testIDs]
        testActuals = [allDataDict[k][targetName] for k in testIDs]

        treeRoot = decisionTree(trainIDs, attrNames)
        globalNB = trainGlobalNaiveBayes(trainIDs, laplace)
        leafModels, leafSizes = buildLeafModels(treeRoot, trainIDs, selK, laplace, smoothC, minSamplesForNB)

        preds, probs = predictWithNBTree(treeRoot, leafModels, testInstances, laplace, smoothC, globalNB)

        for i, p in enumerate(preds):
            if p is None:
                leaf = findLeafNode(treeRoot, testInstances[i])
                lid = id(leaf)
                model = leafModels.get(lid)
                if model is not None and model["nbModel"] is not None:
                    leaf_classes = model["nbModel"]["classes"]
                    leaf_probs = predictNaiveBayesInstance(testInstances[i], model["nbModel"]["classes"], model["nbModel"]["priors"], model["nbModel"]["condProb"], model["nbModel"]["features"], laplace)
                    preds[i] = leaf_classes[np.argmax(leaf_probs)]
                else:
                    class_counts = {}
                    for inst in dataMap.values():
                        c = inst[targetName]
                        class_counts[c] = class_counts.get(c, 0) + 1
                    preds[i] = max(class_counts.items(), key=lambda x: x[1])[0]

        allAcc.append(accuracy_score(testActuals, preds))
        allPrec.append(precision_score(testActuals, preds, average='weighted', zero_division=0))
        allRec.append(recall_score(testActuals, preds, average='weighted', zero_division=0))
        allF1.append(f1_score(testActuals, preds, average='weighted', zero_division=0))
        cm = confusion_matrix(testActuals, preds, labels=allClasses)


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
