from re import I
import pandas as pd
from sklearn import linear_model, model_selection
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
# test
products = '/Users/paddy/Desktop/facebook_products.csv'
images_json = '//Users/paddy/Desktop/images_table 2.json'

class product_cleaner():
    def __init__(self) -> None:
        self.products = products
        self.images = pd.read_json(images_json)
        self.df = pd.read_csv(products)
        self.table_cleaner()
        self.image_classification_encoder()
        self.linear_reg_prep()
        self.linear_regression()
        pass


    def table_cleaner(self):
        df = self.df
        df_images = self.images
        df_images = df_images.rename(columns={'id' : 'image_id', 'product_id' : 'id'})
        merged_df = df_images.merge(df, how = 'inner', on = ['id'])

        merged_df = merged_df.drop(merged_df.columns[4], axis=1)
        merged_df['product_name'] = merged_df['product_name'].str.encode('ascii', 'ignore').str.decode('ascii')
        merged_df['product_description'] = merged_df['product_description'].str.encode('ascii', 'ignore').str.decode('ascii')

        merged_df = merged_df.dropna(axis=0)
        merged_df['price'] = merged_df['price'].str.replace('£','')
        merged_df['price'] = merged_df['price'].str.replace(',','').astype(float)

        merged_df['product_name'] = merged_df['product_name'].str.split('|').str[0]
        merged_df['subcategory'] = merged_df['category'].str.split('/').str[1]
        merged_df['category'] = merged_df['category'].str.split('/').str[0]

        merged_df['county'] = merged_df['location'].str.split(',').str[1]
        merged_df['location'] = merged_df['location'].str.split(',').str[0]     

        merged_df = merged_df[merged_df['price'] < 1000]
        self.merged_df = merged_df[merged_df['price'] > 1]

        return self.merged_df

    def image_classification_encoder(self):
        le = LabelEncoder()
        df_images_final = self.merged_df.drop(['id', 'bucket_link', 'image_ref', 'Unnamed: 0', 'product_name', 'product_description', 'price', 'location', 'url', 'page_id', 'create_time_y', 'subcategory', 'county'], axis=1)
        df_images_final['category'] = le.fit_transform(df_images_final['category'])
        self.df_images_final = df_images_final
        return self.df_images_final


    def linear_reg_prep(self):
        df_for_linear = self.merged_df.drop(['id', 'bucket_link', 'image_ref', 'Unnamed: 0', 'product_name', 'product_description', 'image_id', 'location', 'url', 'page_id', 'create_time_y', 'subcategory', 'county'], axis=1)
        le = LabelEncoder()
        df_for_linear['category'] = le.fit_transform(self.merged_df['category'])
        self.df_for_linear = df_for_linear
        return self.df_for_linear


    def linear_regression(self):
        reg = linear_model.LinearRegression()
        X = self.df_for_linear.drop('price', axis=1)
        y = self.df_for_linear['price']

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
        X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, test_size=0.5)
        X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
        X_test, y_test, test_size=0.5
        )
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        mse_test = mean_squared_error(y[:1529], y_pred)
        print('mse:',mse_test)



if __name__ == '__main__':
    product_cleaner()