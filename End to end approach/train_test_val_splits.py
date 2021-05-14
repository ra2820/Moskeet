import os
import pandas as pd
"""
MbnFilesPath=[]

def listFiles(dir):
	#from pathlib import Path
	global MbnFilesPath;
	for path, subdirs, files in os.walk(dir):
		for name in files:
			#print(os.path.join(path, name))
			n,extension = os.path.splitext(name)
			#print(extension)
			if(extension == '.mbn'):
				MbnFilesPath.append(os.path.join(path, name))

def create_csv(MbnFilesPath): 
    for item in MbnFilesPath:
        data = ""
        data += "{},".format(os.path.basename(item))
        data += "{},".format(item)
        normalized_path = os.path.normpath(item)
        path_components = normalized_path.split(os.sep)
        species = path_components[8]
        data += "{},".format(species)

        file = open(("{}/splitting_data.csv".format(DefaultPath)),"a")
        file.write("\n")
        file.write(data)
        file.close()

DefaultPath = 'C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final\\11. Culiseta incidens\\Cs Incidens' #change path everytime/find a more efficient way
listFiles(DefaultPath)
print(MbnFilesPath)
print(len(MbnFilesPath))
create_csv(MbnFilesPath)
"""



list_csv = []
for path, subdirs, files in os.walk('C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final'):
    for name in files:
        if name == 'splitting_data.csv': 
            os.chdir(path)
            df = pd.read_csv(os.path.join(path, name), header=None)
            df.to_csv("splitting_data.csv", header=["Name", "Path","Species",""], index=False)
            list_csv.append(os.path.join(path, name))

os.chdir('C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final')

#combine all files in the list
combined_split_csv = pd.concat([pd.read_csv(f,header=0) for f in list_csv ])
df_deduplicated = combined_split_csv.drop_duplicates()
df_deduplicated = df_deduplicated.iloc[1:]
#export to csv
df_deduplicated.to_csv( "combined_split_csv.csv", index=False, encoding='utf-8-sig')



os.chdir('C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final')
from sklearn.model_selection import train_test_split
Combined_data = pd.read_csv('C:\\Users\\roopa\\OneDrive\\Desktop\\MOSKEET\\Entire_data_final\\combined_split_csv.csv')
y = Combined_data['Species']
X = Combined_data

X_train_val, X_test, y_train_val, y_test = train_test_split( X, y, test_size=0.15, random_state=42, stratify=y)


X_train,X_val,y_train,y_val = train_test_split(X_train_val,y_train_val,test_size=15/85,random_state=42,stratify=y_train_val)

Combined_data['Set'] = ""
Combined_data.loc[Combined_data.index.isin(y_test.index), 'Set'] = "test"
Combined_data.loc[Combined_data.index.isin(y_train.index), 'Set'] = "train"
Combined_data.loc[Combined_data.index.isin(y_val.index), 'Set'] = "val"

print(Combined_data)

Combined_data.to_csv("combined_split_csv.csv")


#print(y_train)
#print(y_val)
#print(y_test)

