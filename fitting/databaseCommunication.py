#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:51:32 2020

@author: Dypole_Imaging
"""
import numpy as np
import mysql.connector
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


MYSQLserverIP = "192.168.1.133"
username = "root"
password = "w0lfg4ng"
databaseName = "dypoledatabase"

typeOfConnection = "global"
#### Used functions ####

def dataToArray(pathFile):
    # Convert camera fits data to binary format
    with open(pathFile, 'rb') as file:
        image = fits.getdata(file)
    return image[0].ravel().tolist(), image[1].ravel().tolist(), image[2].ravel().tolist()   # atoms, noAtoms, dark

def updateNewImage():
    db = setConnection()
    cursor = db.cursor()
    sql_query = """UPDATE updates SET newImage = 1 WHERE idUpdates = 0;"""
    cursor.execute(sql_query)
    db.commit()
    cursor.close()
    db.close()

def getLastID():
    sql_query = """SELECT runID FROM ciceroOut ORDER BY runID DESC LIMIT 1;"""
    lastRunID = executeGetQuery(sql_query)[0][0]
    sql_query = """SELECT sequenceID FROM sequence ORDER BY sequenceID DESC LIMIT 1;"""
    lastSequenceID = executeGetQuery(sql_query)[0][0]
    return lastRunID, lastSequenceID

def getLastImageID():
    sql_query = """SELECT imageID FROM images ORDER BY imageID DESC LIMIT 1;"""
    lastImageID = executeGetQuery(sql_query)[0][0]
    return lastImageID

def getTimestamp(imageID):
    sql_query = "SELECT timestamp FROM images WHERE imageID = " + str(imageID) + ";"
    timestamp = executeGetQuery(sql_query)[0][0]
    return timestamp

def getLastImageIDs(n):
    sql_query = """SELECT imageID FROM images ORDER BY imageID DESC LIMIT """ + str(n) + """;"""
    lastImageIDsTupleList = executeGetQuery(sql_query)
    lastImageIDs = listTupleToList(lastImageIDsTupleList)
    return lastImageIDs

def listTupleToList(List):
    outputList = []
    for oneTuple in List:
        outputList += [oneTuple[0]]
    return outputList

def setConnection(typeOfConnection = 'global'):
    if typeOfConnection == 'local':
        return setLocalConnection()
    elif typeOfConnection == 'global':
        return setDistantConnection()
    else:
        print('Set what type of connection you want')

def setLocalConnection():
    # Open database connection
    mydb = mysql.connector.connect(host = "localhost",
                        user = "student",
                        password = "w0lfg4ng",
                        database = "imagesdypoledatabase")
    #print('Local connection established')
    return mydb

def setDistantConnection():
    # Open database connection
    mydb = mysql.connector.connect(host = MYSQLserverIP,
                        user = username,
                        password = password,
                        database = databaseName)
    #print('Distant connection established')
    return mydb

def getRunIDFromImageID(imageID):
    sql_query = "SELECT runID_fk FROM images WHERE imageID = {} ;".format(imageID)
    runID_fk = executeGetQuery(sql_query)[0][0]
    return runID_fk

def getNCount(imageID):
    try:
        runID_fk = getRunIDFromImageID(imageID)
        sql_query = "SELECT nCount FROM nCounts WHERE runID_fk = {} ;".format(runID_fk)
        nCount = executeGetQuery(sql_query)[0][0]
    except:
        nCount = 0.0
    return nCount

def executeGetQuery(sql_query): # works when you don't need to use db.commit, so for read only functions
    db = setConnection()
    cursor = db.cursor()
    cursor.execute(sql_query)
    cursorResult = cursor.fetchall()
    cursor.close()
    db.close()
    return cursorResult
    
def saveLocally(nCountArray, holdtimeArray, filename = 'data'):
    filename += '.npy'
    file = open(filename, 'wb')
    np.save(np.array([nCountArray, holdtimeArray]), file, allow_pickle = True)

def getNCountList(imageIDList):
    NCounts = np.zeros(len(imageIDList))
    i = 0
    for imageID in imageIDList:
        NCounts[i] = getNCount(imageID)
        i += 1
    return NCounts

def getHoldTime(imageID):
    runID = getRunIDFromImageID(imageID)
    sql_query = "SELECT BECHoldTime FROM ciceroOut WHERE runID = {} ;".format(runID)
    holdTime = executeGetQuery(sql_query)[0][0]
    return float(holdTime)

def getHoldTimeList(imageIDList):
    holdTimeList = np.zeros(len(imageIDList))
    i = 0
    for imageID in imageIDList:
        holdTimeList[i] = getHoldTime(imageID)
        i += 1
    return holdTimeList

def createDataFrame(imageIDList):
    df_size = len(imageIDList)
    dictionnary = {'imageID' : [], 'BECHoldTime' : [], 'TOF' : [], 'nCount' : []}
    df = pd.DataFrame(dictionnary)
    for imageID in imageIDList:
        runID = getRunIDFromImageID(imageID)
        sql_query = "SELECT BECHoldTime, TOF FROM ciceroOut WHERE runID = {} ;".format(runID)
        BECHoldTime, TOF = executeGetQuery(sql_query)[0][:]
        sql_query = "SELECT nCount FROM nCounts WHERE runID_fk = {} ;".format(runID)
        nCount = executeGetQuery(sql_query)[0][0]
        df = df.append({'imageID' : int(round(imageID)), 'BECHoldTime' : BECHoldTime, 'TOF' : TOF, 'nCount' : nCount}, ignore_index = True)
    return df

def createDataFrame_2(imageIDList):
    df_size = len(imageIDList)
    dictionnary = {'imageID' : [], 'BECHoldTime' : [], 'TOF' : [], 'nCount' : [], 'xWidth' : [], 'yWidth' : [], 'latticeDepth' : [], 'Y_rotation' : []}
    df = pd.DataFrame(dictionnary)
    for imageID in imageIDList:
        runID = getRunIDFromImageID(imageID)
        sql_query = "SELECT BECHoldTime, TOF, latticeDepth, ODT3_Comp, Y_rotation, latticeDepth_final, ODT3_Comp_final FROM ciceroOut WHERE runID = {} ;".format(runID)
        BECHoldTime, TOF, latticeDepth, ODT3_Comp, Y_rotation, latticeDepth_final, ODT3_Comp_final = executeGetQuery(sql_query)[0][:]
        sql_query = "SELECT nCount, xWidth, yWidth FROM nCounts WHERE runID_fk = {} ;".format(runID)
        nCount, xWidth, yWidth = executeGetQuery(sql_query)[0][:] # beware eventually between [:] and [0], make 2 cases
        df = df.append({'imageID' : int(round(imageID)), 'BECHoldTime' : BECHoldTime, 'TOF' : TOF, 'latticeDepth' : latticeDepth, 'ODT3_Comp' : ODT3_Comp, 'Y_rotation' : Y_rotation, 'latticeDepth_final' : latticeDepth_final, 'ODT3_Comp_final' : ODT3_Comp_final, 'nCount' : nCount, 'xWidth' : xWidth, 'yWidth' : yWidth}, ignore_index = True)
    return df

def createDataFrame_list(imageIDList, ciceroVariables, fitVariables = ['nCount', 'xWidth', 'yWidth']):
    df_size = len(imageIDList)
    dictionnary = {}
    for variable in ciceroVariables + fitVariables:
        dictionnary[variable] = []
    df = pd.DataFrame(dictionnary)
    for imageID in imageIDList:
        runID = getRunIDFromImageID(imageID)
        ciceroVariablesString = ' '.join([ciceroVariable for ciceroVariable in ciceroVariables])
        sql_query = "SELECT " + ciceroVariablesString + " FROM ciceroOut WHERE runID = {} ;".format(runID)
        ciceroVariablesOut = executeGetQuery(sql_query)[0][:]
        fitVariablesString = ' '.join([variableFit for variableFit in fitVariables])
        sql_query = "SELECT " + fitVariablesString + " FROM nCounts WHERE runID_fk = {} ;".format(runID)
        fitVariablesOut = executeGetQuery(sql_query)[0][:] # beware eventually between [:] and [0], make 2 cases
        dictTemp = {}
        for ciceroVariable, ciceroVariableOut in zip(ciceroVariables, ciceroVariablesOut):
            dictTemp[ciceroVariable] = ciceroVariableOut
        for fitVariable, fitVariableOut in zip(fitVariables, fitVariablesOut):
            dictTemp[fitVariable] = fitVariableOut
        df = df.append(dictTemp, ignore_index = True)
    return df

def getCameraDimensions(imageID):
    sql_query = "SELECT cameraID_fk FROM images WHERE imageID = " + str(imageID) + ";"
    cameraID = executeGetQuery(sql_query)[0][0]
    sql_query = "SELECT cameraHeight, cameraWidth FROM cameras WHERE cameraID = " + str(cameraID) + ";"
    height, width = executeGetQuery(sql_query)[0]
    return height, width
    
def getImageDatabase(imageID):
    sql_query = """SELECT atoms, noAtoms, dark FROM images WHERE imageID = """ + str(imageID) + """;"""
    byteArrayList = list(executeGetQuery(sql_query)[0])
    i = 0
    for i in range(len(byteArrayList)):
        if type(byteArrayList[i]) == str:
            byteArrayList[i] = bytearray(byteArrayList[i], 'utf-8')
    return byteArrayList  # returns a list of 3 bytearrays

def readDatabaseFile(imageID):
    cameraHeight, cameraWidth = getCameraDimensions(imageID) # look at the config file to make sure there is no conflict zith height and width
    if (cameraHeight == 1 and cameraWidth == 1):
        imageData = [np.zeros(1, dtype = np.uint16).reshape(1,1)]*3
    try:
        byteArrayList = getImageDatabase(imageID)
        # imageData = [np.frombuffer(byteArrayList[0], dtype=np.uint16).reshape(cameraHeight, cameraWidth),
        #              np.frombuffer(byteArrayList[1], dtype=np.uint16).reshape(cameraHeight, cameraWidth),
        #              np.frombuffer(byteArrayList[2], dtype=np.uint16).reshape(cameraHeight, cameraWidth)]
        imageData = np.array([np.frombuffer(byteArrayList[0], dtype=np.uint16).reshape(cameraHeight, cameraWidth),
                     np.frombuffer(byteArrayList[1], dtype=np.uint16).reshape(cameraHeight, cameraWidth),
                     np.frombuffer(byteArrayList[2], dtype=np.uint16).reshape(cameraHeight, cameraWidth)]) # slight difference with the image software version, the output is not a list but a numpy array
    except Exception as e:
        print(str(e))
    return imageData

def createDataFrame_images(imageIDList, crop = [[750, 950], [1150, 1350]]): #crop = ((0, 0), (2000, 1000))):
    df_size = len(imageIDList)
    dictionnary = {'imageID' : [], 'BECHoldTime' : [], 'TOF' : [], 'nCount' : [], 'xWidth' : [], 'yWidth' : [], 'latticeDepth' : [], 'Y_rotation' : []}
    df = pd.DataFrame(dictionnary)
    for imageID in imageIDList:
        runID = getRunIDFromImageID(imageID)
        sql_query = "SELECT BECHoldTime, TOF, latticeDepth, ODT3_Comp, Y_rotation, latticeDepth_final, ODT3_Comp_final FROM ciceroOut WHERE runID = {} ;".format(runID)
        BECHoldTime, TOF, latticeDepth, ODT3_Comp, Y_rotation, latticeDepth_final, ODT3_Comp_final = executeGetQuery(sql_query)[0][:]
        sql_query = "SELECT nCount, xWidth, yWidth FROM nCounts WHERE runID_fk = {} ;".format(runID)
        nCount, xWidth, yWidth = executeGetQuery(sql_query)[0][:] # beware eventually between [:] and [0], make 2 cases
        imageData = readDatabaseFile(imageID)[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        df = df.append({'imageID' : int(round(imageID)), 'BECHoldTime' : BECHoldTime, 'TOF' : TOF, 'latticeDepth' : latticeDepth, 'ODT3_Comp' : ODT3_Comp, 'Y_rotation' : Y_rotation, 'latticeDepth_final' : latticeDepth_final, 'ODT3_Comp_final' : ODT3_Comp_final, 'nCount' : nCount, 'xWidth' : xWidth, 'yWidth' : yWidth, 'imageData' : imageData}, ignore_index = True)
    return df