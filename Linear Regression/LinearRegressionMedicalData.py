import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Read data
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")


# Scale Data
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

ct = make_column_transformer(
    (MinMaxScaler(),['age','bmi','children']),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)

#Encoding Categorical Vars
#insurance = pd.get_dummies(insurance).astype(int)


#Split data



X = insurance.loc[:, insurance.columns != 'charges']
y = insurance['charges']

train_x,test_x,train_y, test_y = train_test_split(X,y, test_size=0.2,random_state=42)

train_x = ct.fit_transform(train_x)
test_x = ct.fit_transform(test_x)




#creating model

model =  tf.keras.Sequential([
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
              metrics=['mae'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.fit(train_x,train_y,epochs=100,callbacks=callback)

pred_y=model.predict(tf.constant(test_x),)



df = pd.DataFrame(list(zip([x for x in range(len(test_y))],test_y,tf.squeeze(pred_y).numpy() )),
               columns =['Ind','Actual', 'Pred'])

print(df)
plt.scatter(df['Ind'],df['Actual'],label="Actual")
plt.scatter(df['Ind'],df['Pred'],label="Pred")

plt.show()










