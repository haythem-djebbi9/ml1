import seaborn as sns
import pandas as pd

# Charger le dataset Titanic depuis seaborn
titanic = sns.load_dataset('titanic')

# Afficher les premières lignes
print(titanic.head(100))

#Afficher les colonnes de dataset
print(f"columns : '{titanic.columns}")


#afficher les type de chaque colonne
print(f"types de colonnes : {titanic.dtypes}")

## Séparer les colonnes en qualitatives et quantitatives
df=pd.DataFrame(titanic)

colonnesQuantitatives=df.select_dtypes(include=['number']).columns
colonnesQualitatives=df.select_dtypes(include=['object','category']).columns

print(f"colonnes quantitatives: {colonnesQuantitatives}")
print(f"colonnes qualitatives: {colonnesQualitatives}")

# Afficher les modalités pour chaque variable catégorielle
for col in colonnesQualitatives:
    print(f"{col}:  {df[col].unique()}")

# Répartition des variables qualitatives
for col in colonnesQualitatives:
    print(f"\nRépartition de la variable '{col}':")
    print(f'{df[col].value_counts()}')

# Description des variables quantitatives
print('Description des variables quantitatives \n')
print(df[colonnesQuantitatives].describe())



# Ce dataset contient-il des valeurs manquantes ? comptez le nombre de valeurs manquantes par colonne
print(f'nombre de valeurs manquantes:{titanic.isnull().sum()} \n')


df1 = titanic.dropna()
print(f"Taille de df1 (sans lignes avec valeurs manquantes) : {df1.shape}")



# Créez un DataFrame `df2` en retirant d'abord la colonne deck, puis en supprimant les observations (lignes) qui contiennent des valeurs manquantes (respectez cet ordre de traitement). Calculez la taille de `df2` et interprétez les résultats.
df2=df.drop(columns='deck').dropna()
print(f'taille de de df2 : {df2.shape}')

#créer un dataframe contenant les variables age et fare
df3=df[['age','fare']]
print('new data frame',df3)

#remplacer les valeurs manquantes de la variable age par sa médiane
ageMedian=df['age'].median()
df['age']=df['age'].fillna(ageMedian)

#importer et instancier MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
scaler = MinMaxScaler()
print(scaler)


#appliquer MinMaxScaler sur df3
df3Scaled = scaler.fit_transform(df3)
df3Scaled = pd.DataFrame(df3Scaled, columns=df3.columns)
print(df3Scaled.head())


# Charger le dataset flights
flights = sns.load_dataset('flights')

# Instanciation du One-Hot Encoder
encoder = OneHotEncoder(sparse_output=False)

# Appliquer One-Hot Encoding sur la colonne 'month'
encoded_months = encoder.fit_transform(flights[['month']])

# Créer un DataFrame des colonnes encodées
encodedDf = pd.DataFrame(encoded_months, columns=encoder.get_feature_names_out(['month']))

# Fusionner les colonnes encodées avec le DataFrame original sans la variable 'month'
dfEncoded = flights.drop('month', axis=1).join(encodedDf)

# Affichage du DataFrame après One-Hot Encoding
print(dfEncoded.head())
