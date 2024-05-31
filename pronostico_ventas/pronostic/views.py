from django.shortcuts import render

from django.shortcuts import render, redirect
from django.http import HttpResponse

import pandas as pd
from django.shortcuts import render
# Create your views here.
#myimports
from .models import CSVFile
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.utils import timezone
from django.shortcuts import render

import numpy as np
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

#index
def index(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('carga_csv')  # Redirige a la página deseada después del inicio de sesión
            else:
                # Las credenciales son incorrectas
                # Puedes mostrar un mensaje de error en la plantilla
                context = {'form': form, 'error_message': 'Credenciales inválidas'}
                return render(request, 'pronostic/index.html', context)
    else:
        form = AuthenticationForm()
    
    context = {'form': form}
    return render(request, 'pronostic/index.html', context)


#vista para crear logoutsrequest
def logout_view(request):
    logout(request)
    return redirect('index') 

#este codigo a continuacion contra la primea view luego del login, este hacia una reg lineal simple

# def carga_csv(request):
#     archivos_csv = CSVFile.objects.all()
#     data = None
#     error_message = None

#     if request.method == 'POST':
#         if 'csv_file' in request.FILES:
#             csv_file = request.FILES['csv_file']
#             if csv_file:
#                 # Guardar el archivo en la base de datos
#                 csv_obj = CSVFile(archivo=csv_file)
#                 csv_obj.save()

#     elif request.method == 'GET' and 'csv_id' in request.GET:
#         csv_id = request.GET['csv_id']
#         try:
#             csv_obj = CSVFile.objects.get(id=csv_id)
#         except CSVFile.DoesNotExist:
#             error_message = 'No se encontró un archivo CSV con el ID proporcionado.'
#             return render(request, 'pronostic/carga_archivo.html', {'data': data, 'archivos_csv': archivos_csv, 'error_message': error_message})

#         try:
#             df = pd.read_csv(csv_obj.archivo.path)  # con el .path para que lea bien

#             # Verificar si la columna 'Kilos' contiene valores de tipo string
#             if df['Kilos'].dtype == 'object':
#                 # Limpiar los valores de la columna 'Kilos'
#                 df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

#             # SOLO DATOS 2022 ENTRENO Filtrar los datos de entrenamiento solo del CSV hasta el año 2022
#             df_train = df[df['Año'] <= 2022]

#             # Definir características (X) y variable objetivo (y) para el entrenamiento
#             X_train = df_train[['Año', 'Mes', 'N° Sku']]
#             y_train = df_train['Kilos']

#             if len(X_train) < 2:
#                 error_message = 'No hay suficientes datos para ajustar un modelo de regresión lineal.'
#                 return render(request, 'pronostic/carga_archivo.html', {'data': data, 'archivos_csv': archivos_csv, 'error_message': error_message})

#             # Inicializar el modelo de regresión lineal
#             modelo_regresion_lineal = LinearRegression()

#             # Ajustar el modelo a los datos de entrenamiento
#             modelo_regresion_lineal.fit(X_train, y_train)

#             # Generar predicciones para cada SKU
#             skus_unicos = df['N° Sku'].unique()
#             # Crear un DataFrame vacío para almacenar los resultados
#             df_predicciones = pd.DataFrame(columns=['N° Sku'] + [f'{mes} 2023' for mes in ['enero', 'febrero', 'marzo', 'abril', 'mayo']])
#             # Agregar los valores reales
#             for sku in skus_unicos:
#                 df_sku_real = df[(df['N° Sku'] == sku) & (df['Año'] == 2023)]
#                 valores_reales = [None] * 5  # Inicializar con None para cada mes
#                 if not df_sku_real.empty:
#                     for index, row in df_sku_real.iterrows():
#                         mes_index = int(row['Mes']) - 1  # Convertir a entero para obtener el índice correcto del mes
#                         valores_reales[mes_index] = row['Kilos']  # Asignar el valor real al mes correspondiente
#                     # Convertir valores_reales a un diccionario con las claves correspondientes
#                     valores_reales_dict = {f'{mes} 2023': valor for mes, valor in zip(['enero', 'febrero', 'marzo', 'abril', 'mayo'], valores_reales)}
#                     valores_reales_dict['N° Sku'] = f'{sku} Real'  # Agregar el número de SKU al diccionario
#                     df_predicciones = df_predicciones.append(valores_reales_dict, ignore_index=True)
        
#             # Agregar las predicciones
#             for sku in skus_unicos:
#                 df_sku_predicciones = pd.DataFrame({'N° Sku': [sku]})
#                 for i, mes in enumerate(['enero', 'febrero', 'marzo', 'abril', 'mayo'], 1):  # Iterar sobre los meses de enero a mayo
#                     df_sku = pd.DataFrame({'Año': [2023], 'Mes': [i], 'N° Sku': [sku]})
#                     if not df_sku_real.empty:
#                         prediccion_sku = modelo_regresion_lineal.predict(df_sku[['Año', 'Mes', 'N° Sku']])
#                         df_sku_predicciones[f'{mes} 2023'] = prediccion_sku[0]  # Asignar la predicción al mes correspondiente
#                     else:
#                         df_sku_predicciones[f'{mes} 2023'] = np.nan  # Si no hay datos, llenar con NaN
#                 df_predicciones = df_predicciones.append(df_sku_predicciones, ignore_index=True)





    #         # Eliminar la última fila vacía si no contiene ningún valor
    #         if df_predicciones.iloc[-1].isnull().all():
    #             df_predicciones = df_predicciones[:-1]

    #         # Convertir DataFrame de predicciones a formato HTML
    #         data = df_predicciones.to_html(index=False)

    #     except Exception as e:
    #         error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"
    #         return render(request, 'pronostic/carga_archivo.html', {'data': data, 'archivos_csv': archivos_csv, 'error_message': error_message})

    # return render(request, 'pronostic/carga_archivo.html', {'data': data, 'archivos_csv': archivos_csv, 'error_message': error_message})



import pandas as pd
from sklearn.linear_model import LinearRegression

def carga_csv(request):
    error_message = None

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            if csv_file:
                try:
                    # Leer el archivo CSV subido
                    df = pd.read_csv(csv_file)

                    # Verificar si la columna 'Kilos' contiene valores de tipo string
                    if df['Kilos'].dtype == 'object':
                        # Limpiar los valores de la columna 'Kilos'
                        df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

                    # SOLO DATOS 2022 ENTRENO: Filtrar los datos de entrenamiento solo del CSV hasta el año 2022
                    df_train = df[df['Año'] <= 2022]

                    # Definir características (X) y variable objetivo (y) para el entrenamiento
                    X_train = df_train[['Año', 'Mes', 'N° Sku']]
                    y_train = df_train['Kilos']

                    if len(X_train) < 2:
                        error_message = 'No hay suficientes datos para ajustar un modelo de regresión lineal.'
                        return render(request, 'pronostic/carga_archivo.html', {'error_message': error_message})

                    # Inicializar el modelo de regresión lineal
                    modelo_regresion_lineal = LinearRegression()

                    # Ajustar el modelo a los datos de entrenamiento
                    modelo_regresion_lineal.fit(X_train, y_train)

                    # Generar predicciones para cada SKU y el mes seleccionado
                    mes_seleccionado = int(request.POST.get('mes', 1))  # Obtener el mes seleccionado, por defecto es enero
                    skus_unicos = df['N° Sku'].unique()
                    df_predicciones = pd.DataFrame(columns=['Año', 'Mes', 'N° Sku', 'Predicciones'])

                    for sku in skus_unicos:
                        df_sku = pd.DataFrame({'Año': [2023], 'Mes': [mes_seleccionado], 'N° Sku': [sku]})
                        prediccion_sku = modelo_regresion_lineal.predict(df_sku[['Año', 'Mes', 'N° Sku']])
                        df_sku['Predicciones'] = prediccion_sku
                        df_predicciones = pd.concat([df_predicciones, df_sku], ignore_index=True)

                    # Convertir DataFrame de predicciones a formato HTML
                    data = df_predicciones.to_html(index=False)

                    return render(request, 'pronostic/carga_archivo.html', {'data': data})

                except Exception as e:
                    error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"
                    return render(request, 'pronostic/carga_archivo.html', {'error_message': error_message})

    return render(request, 'pronostic/carga_archivo.html', {'error_message': error_message})



#d
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from django.shortcuts import render
from django.conf import settings
from sklearn.linear_model import LinearRegression
from .models import CSVFile

def graficos(request):
    archivos_csv = CSVFile.objects.all()
    plot_path = None
    error_message = None

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            if csv_file:
                csv_obj = CSVFile(archivo=csv_file)
                csv_obj.save()

    elif request.method == 'GET' and 'csv_id' in request.GET:
        csv_id = request.GET['csv_id']
        try:
            csv_obj = CSVFile.objects.get(id=csv_id)
        except CSVFile.DoesNotExist:
            error_message = 'No se encontró un archivo CSV con el ID proporcionado.'
            return render(request, 'pronostic/graficos.html', {'plot_path': plot_path, 'archivos_csv': archivos_csv, 'error_message': error_message})

        try:
            df = pd.read_csv(csv_obj.archivo.path)

            if df['Kilos'].dtype == 'object':
                df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

            df_train = df[df['Año'] <= 2022]
            X_train = df_train[['Año', 'Mes', 'N° Sku']]
            y_train = df_train['Kilos']

            if len(X_train) < 2:
                error_message = 'No hay suficientes datos para ajustar un modelo de regresión lineal.'
                return render(request, 'pronostic/graficos.html', {'plot_path': plot_path, 'archivos_csv': archivos_csv, 'error_message': error_message})

            modelo_regresion_lineal = LinearRegression()
            modelo_regresion_lineal.fit(X_train, y_train)

            mes_seleccionado = int(request.GET.get('mes', 1))
            skus_unicos = df['N° Sku'].unique()
            plt.figure(figsize=(10, 6))
            for sku in skus_unicos:
                df_sku = pd.DataFrame({'Año': [2023], 'Mes': [mes_seleccionado], 'N° Sku': [sku]})
                prediccion_sku = modelo_regresion_lineal.predict(df_sku[['Año', 'Mes', 'N° Sku']])
                plt.scatter(df_sku['Año'], prediccion_sku, label=f'SKU {sku} Predicción')
            plt.xlabel('Año')
            plt.ylabel('Kilos')
            plt.title('Predicciones de Regresión Lineal Simple')
            plt.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Define the static path and create the directory if it does not exist
            static_dir = '/home/renzorios/forecast_projects/pronostico_ventas/pronostic/static'
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                
            plot_path = os.path.join(static_dir, 'grafico.png')
            with open(plot_path, 'wb') as f:
                f.write(buffer.getvalue())
            buffer.close()
            plt.close()

            plot_path = '/static/grafico.png'

        except Exception as e:
            error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"
            return render(request, 'pronostic/graficos.html', {'plot_path': plot_path, 'archivos_csv': archivos_csv, 'error_message': error_message})

    return render(request, 'pronostic/graficos.html', {'plot_path': plot_path, 'archivos_csv': archivos_csv, 'error_message': error_message})








import pandas as pd
from django.shortcuts import render
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import io
import base64

def decision_tree(request):
    data = []
    error_message = None

    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

        if csv_file:
            try:
                # Leer el archivo CSV
                df = pd.read_csv(csv_file)

                # Verificar si la columna 'Kilos' contiene valores de tipo string
                if df['Kilos'].dtype == 'object':
                    df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

                # Verifica si hay valores negativos o cero y ajusta según sea necesario
                if (df['Kilos'] <= 0).any():
                    error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
                else:
                    # Iterar sobre cada SKU en el DataFrame
                    for sku in df['N° Sku'].unique():
                        df_sku = df[df['N° Sku'] == sku]

                        # Inicializa un DataFrame para almacenar las predicciones
                        forecast_2023 = pd.DataFrame(columns=['Año', 'Mes', 'Predicción'])

                        # Usar los años y meses como características (X) y Kilos como variable dependiente (y)
                        X = df_sku[['Año', 'Mes']]
                        y = df_sku['Kilos']

                        # Transformar las características para la regresión
                        X['Mes'] = X['Mes'].apply(lambda x: (x - 1) / 11)  # Normalizar el mes

                        # Iterar sobre cada mes de 2023
                        for month in range(1, 13):
                            # Filtrar los datos para el mes actual
                            X_month = X[X['Mes'] == (month - 1) / 11]
                            y_month = y[X['Mes'] == (month - 1) / 11]

                            # Inicializa el modelo de árbol de decisión
                            model = DecisionTreeRegressor()

                            # Ajusta el modelo a los datos del mes actual
                            model.fit(X_month, y_month)

                            # Generar predicciones para el mes actual
                            X_pred = pd.DataFrame({
                                'Año': [2023],
                                'Mes': [(month - 1) / 11]  # Normalizar el mes
                            })
                            y_pred = model.predict(X_pred)

                            # Agregar la predicción al DataFrame
                            forecast_2023 = forecast_2023.append({
                                'Año': 2023,
                                'Mes': month,
                                'Predicción': y_pred[0],
                            }, ignore_index=True)

                        # Convertir la tabla de predicciones a HTML y almacenar en el diccionario
                        table_html = forecast_2023.to_html(index=False)

                        # Generar el gráfico
                        plt.figure()
                        plt.plot(forecast_2023['Mes'], forecast_2023['Predicción'], marker='o')
                        plt.title(f'Predicción de Kilos para SKU {sku} en 2023')
                        plt.xlabel('Mes')
                        plt.ylabel('Predicción de Kilos')
                        plt.grid(True)

                        # Guardar el gráfico en un objeto BytesIO
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        string = base64.b64encode(buf.read())
                        uri = 'data:image/png;base64,' + string.decode('utf-8')

                        # Almacenar la tabla y el gráfico en una lista
                        data.append({'sku': sku, 'table_html': table_html, 'graph': uri})

            except Exception as e:
                error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

    return render(request, 'pronostic/decision_tree.html', {'data': data, 'error_message': error_message})




import pandas as pd
from django.shortcuts import render
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64

def random_forest(request):
    data = []
    error_message = None

    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

        if csv_file:
            try:
                # Leer el archivo CSV
                df = pd.read_csv(csv_file)

                # Verificar si la columna 'Kilos' contiene valores de tipo string
                if df['Kilos'].dtype == 'object':
                    df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

                # Verifica si hay valores negativos o cero y ajusta según sea necesario
                if (df['Kilos'] <= 0).any():
                    error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
                else:
                    # Iterar sobre cada SKU en el DataFrame
                    for sku in df['N° Sku'].unique():
                        df_sku = df[df['N° Sku'] == sku]

                        # Inicializa un DataFrame para almacenar las predicciones
                        forecast_2023 = pd.DataFrame(columns=['Año', 'Mes', 'Predicción'])

                        # Usar los años y meses como características (X) y Kilos como variable dependiente (y)
                        X = df_sku[['Año', 'Mes']]
                        y = df_sku['Kilos']

                        # Transformar las características para la regresión
                        X['Mes'] = X['Mes'].apply(lambda x: (x - 1) / 11)  # Normalizar el mes

                        # Iterar sobre cada mes de 2023
                        for month in range(1, 13):
                            # Filtrar los datos para el mes actual
                            X_month = X[X['Mes'] == (month - 1) / 11]
                            y_month = y[X['Mes'] == (month - 1) / 11]

                            # Inicializa el modelo de Random Forest
                            model = RandomForestRegressor()

                            # Ajusta el modelo a los datos del mes actual
                            model.fit(X_month, y_month)

                            # Generar predicciones para el mes actual
                            X_pred = pd.DataFrame({
                                'Año': [2023],
                                'Mes': [(month - 1) / 11]  # Normalizar el mes
                            })
                            y_pred = model.predict(X_pred)

                            # Agregar la predicción al DataFrame
                            forecast_2023 = forecast_2023.append({
                                'Año': 2023,
                                'Mes': month,
                                'Predicción': y_pred[0],
                            }, ignore_index=True)

                        # Convertir la tabla de predicciones a HTML y almacenar en el diccionario
                        table_html = forecast_2023.to_html(index=False)

                        # Generar el gráfico
                        plt.figure()
                        plt.plot(forecast_2023['Mes'], forecast_2023['Predicción'], marker='o')
                        plt.title(f'Predicción de Kilos para SKU {sku} en 2023')
                        plt.xlabel('Mes')
                        plt.ylabel('Predicción de Kilos')
                        plt.grid(True)

                        # Guardar el gráfico en un objeto BytesIO
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        string = base64.b64encode(buf.read())
                        uri = 'data:image/png;base64,' + string.decode('utf-8')

                        # Almacenar la tabla y el gráfico en una lista
                        data.append({'sku': sku, 'table_html': table_html, 'graph': uri})

            except Exception as e:
                error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

    return render(request, 'pronostic/randomforest.html', {'data': data, 'error_message': error_message})





import pandas as pd
from django.shortcuts import render
import xgboost as xgb
import matplotlib.pyplot as plt
import io
import base64

def xgboost(request):
    data = []
    error_message = None

    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

        if csv_file:
            try:
                # Leer el archivo CSV
                df = pd.read_csv(csv_file)

                # Verificar si la columna 'Kilos' contiene valores de tipo string
                if df['Kilos'].dtype == 'object':
                    df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

                # Verifica si hay valores negativos o cero y ajusta según sea necesario
                if (df['Kilos'] <= 0).any():
                    error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
                else:
                    # Iterar sobre cada SKU en el DataFrame
                    for sku in df['N° Sku'].unique():
                        df_sku = df[df['N° Sku'] == sku]

                        # Inicializa un DataFrame para almacenar las predicciones
                        forecast_2023 = pd.DataFrame(columns=['Año', 'Mes', 'Predicción'])

                        # Usar los años y meses como características (X) y Kilos como variable dependiente (y)
                        X = df_sku[['Año', 'Mes']]
                        y = df_sku['Kilos']

                        # Transformar las características para la regresión
                        X['Mes'] = X['Mes'].apply(lambda x: (x - 1) / 11)  # Normalizar el mes

                        # Iterar sobre cada mes de 2023
                        for month in range(1, 13):
                            # Filtrar los datos para el mes actual
                            X_month = X[X['Mes'] == (month - 1) / 11]
                            y_month = y[X['Mes'] == (month - 1) / 11]

                            # Convertir los datos a una matriz DMatrix de XGBoost
                            dmatrix = xgb.DMatrix(X_month, label=y_month)

                            # Definir los parámetros del modelo XGBoost
                            params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}

                            # Entrenar el modelo XGBoost
                            model = xgb.train(params, dmatrix, num_boost_round=100)

                            # Generar predicciones para el mes actual
                            X_pred = pd.DataFrame({
                                'Año': [2023],
                                'Mes': [(month - 1) / 11]  # Normalizar el mes
                            })
                            dmatrix_pred = xgb.DMatrix(X_pred)
                            y_pred = model.predict(dmatrix_pred)

                            # Agregar la predicción al DataFrame
                            forecast_2023 = forecast_2023.append({
                                'Año': 2023,
                                'Mes': month,
                                'Predicción': y_pred[0],
                            }, ignore_index=True)

                        # Convertir la tabla de predicciones a HTML y almacenar en el diccionario
                        table_html = forecast_2023.to_html(index=False)

                        # Generar el gráfico
                        plt.figure()
                        plt.plot(forecast_2023['Mes'], forecast_2023['Predicción'], marker='o')
                        plt.title(f'Predicción de Kilos para SKU {sku} en 2023')
                        plt.xlabel('Mes')
                        plt.ylabel('Predicción de Kilos')
                        plt.grid(True)

                        # Guardar el gráfico en un objeto BytesIO
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        string = base64.b64encode(buf.read())
                        uri = 'data:image/png;base64,' + string.decode('utf-8')

                        # Almacenar la tabla y el gráfico en una lista
                        data.append({'sku': sku, 'table_html': table_html, 'graph': uri})

            except Exception as e:
                error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

    return render(request, 'pronostic/xgboost.html', {'data': data, 'error_message': error_message})







import pandas as pd
from django.conf import settings
from django.shortcuts import render
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from .models import CSVFile

def graficos_tree(request):
    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            df = pd.read_csv(csv_file)

            if 'Kilos' in df.columns and df['Kilos'].dtype == 'O':
                df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)
            else:
                return render(request, 'pronostic/tree_graficos.html', {'error_message': 'La columna Kilos no está presente o no es del tipo adecuado'})

            sku = request.POST.get('sku_manual')
            if sku:
                df_sku = df[df['N° Sku'] == int(sku)]
            else:
                return render(request, 'pronostic/tree_graficos.html', {'error_message': 'Debe ingresar un SKU válido'})

            df_sku['Meses'] = (df_sku['Año'] - df_sku['Año'].min()) * 12 + df_sku['Mes'] - 1
            X = df_sku[['Meses']]
            y = df_sku['Kilos']

            modelo = DecisionTreeRegressor()
            modelo.fit(X, y)

            X_2023 = pd.DataFrame({'Meses': [(2023 - df_sku['Año'].min()) * 12 + m for m in range(1, 13)]})
            y_2023_pred = modelo.predict(X_2023)
            predicciones_2023 = pd.DataFrame({'Año': [2023]*12, 'Mes': range(1, 13), 'Kilos': y_2023_pred})
            df_sku = pd.concat([df_sku, predicciones_2023])

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Define the static path and create the directory if it does not exist
            static_dir = '/home/renzorios/forecast_projects/pronostico_ventas/pronostic/static'
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                
            plot_path = os.path.join(static_dir, 'grafico.png')
            with open(plot_path, 'wb') as f:
                f.write(buffer.getvalue())
            buffer.close()
            plt.close()

            plot_path = '/static/grafico.png'


            # Guarda el gráfico en la ruta especificada
            plt.figure(figsize=(10, 6))
            plt.plot(df_sku['Año'] + df_sku['Mes'] / 12, df_sku['Kilos'], label='Datos históricos y predicciones')
            plt.xlabel('Año')
            plt.ylabel('Kilos')
            plt.title(f'Pronóstico de Kilos para SKU {sku} a través del Tiempo')
            plt.legend()
            plt.grid(True)
            plt.savefig(plot_path)
            plt.close()

            return render(request, 'pronostic/tree_graficos.html', {'plot_path': plot_path})

    archivos_csv = CSVFile.objects.all()  # Asegúrate de obtener tus archivos CSV desde la base de datos aquí
    return render(request, 'pronostic/tree_graficos.html', {'archivos_csv': archivos_csv})

# mi_app/views.py










#este codigo no itera por meses, predice a partir de todos los datos

# import pandas as pd
# import io
# import base64
# import matplotlib.pyplot as plt
# from django.shortcuts import render
# from sklearn.linear_model import LinearRegression

# def profeta_vista(request):
#     data = None
#     image_base64 = None
#     error_message = None

#     if request.method == 'POST' and 'csv_file' in request.FILES:
#         csv_file = request.FILES['csv_file']
        
#         if csv_file:
#             try:
#                 # Leer el archivo CSV
#                 df = pd.read_csv(csv_file)

#                 # Verificar si la columna 'Kilos' contiene valores de tipo string
#                 if df['Kilos'].dtype == 'object':
#                     df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

#                 # Verifica si hay valores negativos o cero y ajusta según sea necesario
#                 if (df['Kilos'] <= 0).any():
#                     error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
#                 else:
#                     # Crea una nueva columna 'ds' que contiene la fecha en formato YYYY-MM-DD
#                     df['ds'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')

#                     # Usar los años como característica (X) y Kilos como variable dependiente (y)
#                     X = df['ds'].dt.year.values.reshape(-1, 1)
#                     y = df['Kilos'].values

#                     # Inicializa el modelo de regresión lineal
#                     model = LinearRegression()

#                     # Ajusta el modelo a los datos
#                     model.fit(X, y)

#                     # Predicción para el año 2023
#                     pred_year = 2023
#                     y_pred = model.predict([[pred_year]])

#                     # Generar datos de predicción para graficar
#                     future_years = list(range(df['ds'].dt.year.min(), 2024))
#                     future_preds = model.predict([[year] for year in future_years])

#                     # Graficar los resultados y guardar la imagen en un buffer
#                     plt.figure(figsize=(10, 6))
#                     plt.scatter(X, y, color='blue', label='Datos históricos')
#                     plt.plot(future_years, future_preds, color='red', linestyle='--', label='Predicción')
#                     plt.xlabel('Año')
#                     plt.ylabel('Kilos')
#                     plt.legend()
#                     plt.title('Predicción de Kilos usando Regresión Lineal')

#                     buf = io.BytesIO()
#                     plt.savefig(buf, format='png')
#                     buf.seek(0)
#                     image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#                     plt.close()

#                     # Crear una tabla con la predicción para el año 2023
#                     forecast_2023 = pd.DataFrame({
#                         'ds': [pd.to_datetime(f'{pred_year}')],
#                         'yhat': y_pred,
#                     })

#                     # Convertir la tabla a HTML
#                     data = forecast_2023.to_html(index=False)

#             except Exception as e:
#                 error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

#     return render(request, 'pronostic/profeta.html', {'data': data, 'error_message': error_message, 'image_base64': image_base64})





import pandas as pd
from prophet import Prophet
from django.shortcuts import render
import matplotlib.pyplot as plt
import tempfile
import base64

def profeta_vista(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']

        # Leer el archivo CSV
        df = pd.read_csv(csv_file)

        # Seleccionar el SKU deseado (ejemplo: 10000027)
        sku_deseado = 10000027
        df_sku = df[df['N° Sku'] == sku_deseado]

        # Convertir la fecha a formato datetime
        df_sku['ds'] = pd.to_datetime(df_sku['Año'].astype(str) + '-' + df_sku['Mes'].astype(str) + '-01')

        # Renombrar la columna de kilos a 'y'
        df_sku = df_sku.rename(columns={'Kilos': 'y'})

        # Establecer un límite inferior (floor) de 0
        df_sku['floor'] = 0

        # Seleccionar solo los datos hasta 2022
        df_train = df_sku[df_sku['ds'].dt.year <= 2022]

        # Inicializar y ajustar el modelo con regresión lineal
        m = Prophet(growth='linear')
        m.add_seasonality(name='linear', period=365.25/12, fourier_order=1)
        m.fit(df_train)

        # Definir la fecha de predicción para enero de 2023
        future = pd.DataFrame({'ds': ['2023-01-01'], 'floor': [0]})

        # Realizar la predicción para enero de 2023
        forecast = m.predict(future)

        # Obtener solo la predicción para enero de 2023
        prediction_jan_2023 = forecast.loc[0, 'yhat']

        # Asegurarse de que la predicción no sea negativa
        prediction_jan_2023 = max(prediction_jan_2023, 0)

        # Datos para mostrar en la tabla
        data = forecast[['ds', 'yhat']].to_html(index=False)

        # Crear un archivo temporal para guardar la gráfica
        with tempfile.NamedTemporaryFile(suffix='.png') as tmpfile:
            fig = m.plot(forecast)
            plt.title('Predicción para Enero 2023')
            plt.xlabel('Fecha')
            plt.ylabel('Kilos')
            plt.savefig(tmpfile.name, format='png')
            
            # Leer el archivo temporal y convertirlo a base64
            tmpfile.seek(0)
            image_base64 = base64.b64encode(tmpfile.read()).decode('utf-8')

        return render(request, 'pronostic/profeta.html', {'image_base64': image_base64, 'data': data})
    else:
        return render(request, 'pronostic/profeta.html')











# import calendar
# import pandas as pd
# import io
# import base64
# import matplotlib.pyplot as plt
# from django.shortcuts import render
# from sklearn.linear_model import LinearRegression

# def profeta_tabla(request):
#     data = None
#     image_base64 = None
#     error_message = None
#     selected_month = None  # Agregamos esta línea para asegurarnos de que la variable se defina

#     if request.method == 'POST' and 'csv_file' in request.FILES:
#         csv_file = request.FILES['csv_file']
#         selected_month = request.POST.get('month')

#         if csv_file and selected_month:
#             try:
#                 # Leer el archivo CSV
#                 df = pd.read_csv(csv_file)

#                 # Verificar si la columna 'Kilos' contiene valores de tipo string
#                 if df['Kilos'].dtype == 'object':
#                     df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

#                 # Verifica si hay valores negativos o cero y ajusta según sea necesario
#                 if (df['Kilos'] <= 0).any():
#                     error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
#                 else:
#                     # Filtrar el DataFrame por el mes seleccionado
#                     df_selected_month = df[df['Mes'] == int(selected_month)]

#                     # Usar los años como característica (X) y Kilos como variable dependiente (y)
#                     X = df_selected_month['Año'].values.reshape(-1, 1)
#                     y = df_selected_month['Kilos'].values

#                     # Inicializa el modelo de regresión lineal
#                     model = LinearRegression()

#                     # Ajusta el modelo a los datos del mes seleccionado
#                     model.fit(X, y)

#                     # Predicción para el año 2023
#                     pred_year = 2023
#                     y_pred = model.predict([[pred_year]])

#                     # Graficar los resultados y guardar la imagen en un buffer
#                     plt.figure(figsize=(10, 6))
#                     plt.scatter(X, y, color='blue', label='Datos históricos')
#                     plt.plot([pred_year], y_pred, 'ro', label='Predicción')
#                     plt.xlabel('Año')
#                     plt.ylabel('Kilos')
#                     plt.legend()
#                     plt.title('Predicción de Kilos para el mes seleccionado en 2023')

#                     buf = io.BytesIO()
#                     plt.savefig(buf, format='png')
#                     buf.seek(0)
#                     image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#                     plt.close()

#                     # Crear una tabla con la predicción para el año 2023
#                     forecast_2023 = pd.DataFrame({
#                         'Año': [pred_year],
#                         'Mes': [int(selected_month)],
#                         'Prediccion': y_pred,
#                     })

#                     # Convertir la tabla a HTML
#                     data = forecast_2023.to_html(index=False)

#             except Exception as e:
#                 error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

#     months = [(i, calendar.month_name[i]) for i in range(1, 13)]  # Asegurémonos de definir esta variable

#     return render(request, 'pronostic/profeta_tabla.html', {'data': data, 'error_message': error_message, 'image_base64': image_base64, 'months': months, 'selected_month': selected_month})  # Añadimos 'months' y 'selected_month' al contexto















import pandas as pd
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def profeta_tabla(request):
    data = []
    error_message = None

    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

        if csv_file:
            try:
                # Leer el archivo CSV
                df = pd.read_csv(csv_file)

                # Verificar si la columna 'Kilos' contiene valores de tipo string
                if df['Kilos'].dtype == 'object':
                    df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

                # Verifica si hay valores negativos o cero y ajusta según sea necesario
                if (df['Kilos'] <= 0).any():
                    error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
                else:
                    # Iterar sobre cada SKU en el DataFrame
                    for sku in df['N° Sku'].unique():
                        df_sku = df[df['N° Sku'] == sku]

                        # Inicializa un DataFrame para almacenar las predicciones
                        forecast_2023 = pd.DataFrame(columns=['Año', 'Mes', 'Predicción'])

                        # Usar los años y meses como características (X) y Kilos como variable dependiente (y)
                        X = df_sku[['Año', 'Mes']]
                        y = df_sku['Kilos']

                        # Transformar las características para la regresión
                        X['Mes'] = X['Mes'].apply(lambda x: (x - 1) / 11)  # Normalizar el mes

                        # Listas para almacenar predicciones reales y predichas
                        y_true = []
                        y_pred = []

                        # Iterar sobre cada mes de 2023
                        for month in range(1, 13):
                            # Filtrar los datos para el mes actual
                            X_month = X[X['Mes'] == (month - 1) / 11]
                            y_month = y[X['Mes'] == (month - 1) / 11]

                            # Inicializa el modelo de regresión lineal
                            model = LinearRegression()

                            # Ajusta el modelo a los datos del mes actual
                            model.fit(X_month, y_month)

                            # Generar predicciones para el mes actual
                            X_pred = pd.DataFrame({
                                'Año': [2023],
                                'Mes': [(month - 1) / 11]  # Normalizar el mes
                            })
                            y_pred_month = model.predict(X_pred)

                            # Agregar la predicción al DataFrame
                            forecast_2023 = forecast_2023.append({
                                'Año': 2023,
                                'Mes': month,
                                'Predicción': y_pred_month[0],
                            }, ignore_index=True)

                            # Almacenar valores reales y predichos para evaluación
                            y_true.extend(y_month.values)
                            y_pred.extend(model.predict(X_month))

                        # Cálculos de evaluación
                        r2 = r2_score(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_true, y_pred)

                        # Convertir la tabla de predicciones a HTML y almacenar en el diccionario
                        table_html = forecast_2023.to_html(index=False)

                        # Generar el gráfico
                        plt.figure()
                        plt.plot(forecast_2023['Mes'], forecast_2023['Predicción'], marker='o')
                        plt.title(f'Predicción de Kilos para SKU {sku} en 2023')
                        plt.xlabel('Mes')
                        plt.ylabel('Predicción de Kilos')
                        plt.grid(True)

                        # Guardar el gráfico en un objeto BytesIO
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        string = base64.b64encode(buf.read())
                        uri = 'data:image/png;base64,' + string.decode('utf-8')

                        # Almacenar la tabla, el gráfico y las métricas en una lista
                        data.append({
                            'sku': sku,
                            'table_html': table_html,
                            'graph': uri,
                            'r2': r2,
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae
                        })

            except Exception as e:
                error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

    return render(request, 'pronostic/profeta_tabla.html', {'data': data, 'error_message': error_message})





############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


import pandas as pd
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64

def dashboard(request):
    data = []
    error_message = None

    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

        if csv_file:
            try:
                # Leer el archivo CSV
                df = pd.read_csv(csv_file)

                # Verificar si la columna 'Kilos' contiene valores de tipo string
                if df['Kilos'].dtype == 'object':
                    df['Kilos'] = df['Kilos'].str.replace(',', '').astype(float)

                # Verifica si hay valores negativos o cero y ajusta según sea necesario
                if (df['Kilos'] <= 0).any():
                    error_message = "Los datos de 'Kilos' contienen valores negativos o cero que no son válidos."
                else:
                    # Iterar sobre cada SKU en el DataFrame
                    for sku in df['N° Sku'].unique():
                        df_sku = df[df['N° Sku'] == sku]

                        # Inicializa un DataFrame para almacenar las predicciones
                        forecast_2023 = pd.DataFrame(columns=['Año', 'Mes', 'Linear_Pred', 'DecisionTree_Pred', 'RandomForest_Pred', 'XGBoost_Pred'])

                        # Usar los años y meses como características (X) y Kilos como variable dependiente (y)
                        X = df_sku[['Año', 'Mes']]
                        y = df_sku['Kilos']

                        # Transformar las características para la regresión
                        X['Mes'] = X['Mes'].apply(lambda x: (x - 1) / 11)  # Normalizar el mes

                        # Listas para almacenar predicciones reales y predichas para cada modelo
                        y_true = {'Linear': [], 'DecisionTree': [], 'RandomForest': [], 'XGBoost': []}
                        y_pred = {'Linear': [], 'DecisionTree': [], 'RandomForest': [], 'XGBoost': []}

                        # Inicializar los modelos
                        linear_model = LinearRegression()
                        decision_tree_model = DecisionTreeRegressor()
                        random_forest_model = RandomForestRegressor(n_estimators=100)
                        xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

                        # Iterar sobre cada mes de 2023
                        for month in range(1, 13):
                            # Filtrar los datos para el mes actual
                            X_month = X[X['Mes'] == (month - 1) / 11]
                            y_month = y[X['Mes'] == (month - 1) / 11]

                            # Ajustar los modelos a los datos del mes actual
                            linear_model.fit(X_month, y_month)
                            decision_tree_model.fit(X_month, y_month)
                            random_forest_model.fit(X_month, y_month)
                            xgboost_model.fit(X_month, y_month)

                            # Generar predicciones para el mes actual
                            X_pred = pd.DataFrame({
                                'Año': [2023],
                                'Mes': [(month - 1) / 11]  # Normalizar el mes
                            })
                            linear_pred = linear_model.predict(X_pred)[0]
                            decision_tree_pred = decision_tree_model.predict(X_pred)[0]
                            random_forest_pred = random_forest_model.predict(X_pred)[0]
                            xgboost_pred = xgboost_model.predict(X_pred)[0]

                            # Agregar las predicciones al DataFrame
                            forecast_2023 = forecast_2023.append({
                                'Año': 2023,
                                'Mes': month,
                                'Linear_Pred': linear_pred,
                                'DecisionTree_Pred': decision_tree_pred,
                                'RandomForest_Pred': random_forest_pred,
                                'XGBoost_Pred': xgboost_pred,
                            }, ignore_index=True)

                            # Almacenar valores reales y predichos para evaluación
                            y_true['Linear'].extend(y_month.values)
                            y_true['DecisionTree'].extend(y_month.values)
                            y_true['RandomForest'].extend(y_month.values)
                            y_true['XGBoost'].extend(y_month.values)
                            
                            y_pred['Linear'].extend(linear_model.predict(X_month))
                            y_pred['DecisionTree'].extend(decision_tree_model.predict(X_month))
                            y_pred['RandomForest'].extend(random_forest_model.predict(X_month))
                            y_pred['XGBoost'].extend(xgboost_model.predict(X_month))

                        # Convertir la tabla de predicciones a HTML y almacenar en el diccionario
                        table_html = forecast_2023.to_html(index=False)

                        # Generar el gráfico
                        plt.figure()
                        plt.plot(forecast_2023['Mes'], forecast_2023['Linear_Pred'], marker='o', label='Linear Regression')
                        plt.plot(forecast_2023['Mes'], forecast_2023['DecisionTree_Pred'], marker='o', label='Decision Tree')
                        plt.plot(forecast_2023['Mes'], forecast_2023['RandomForest_Pred'], marker='o', label='Random Forest')
                        plt.plot(forecast_2023['Mes'], forecast_2023['XGBoost_Pred'], marker='o', label='XGBoost')
                        plt.title(f'Predicción de Kilos para SKU {sku} en 2023')
                        plt.xlabel('Mes')
                        plt.ylabel('Predicción de Kilos')
                        plt.legend()
                        plt.grid(True)

                        # Guardar el gráfico en un objeto BytesIO
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        string = base64.b64encode(buf.read())
                        uri = 'data:image/png;base64,' + string.decode('utf-8')

                        # Calcular las métricas de evaluación para cada modelo
                        metrics = {
                            'Linear': {
                                'R2': r2_score(y_true['Linear'], y_pred['Linear']),
                                'RMSE': mean_squared_error(y_true['Linear'], y_pred['Linear'], squared=False),
                                'MAE': mean_absolute_error(y_true['Linear'], y_pred['Linear']),
                                'MSE': mean_squared_error(y_true['Linear'], y_pred['Linear'])
                            },
                            'DecisionTree': {
                                'R2': r2_score(y_true['DecisionTree'], y_pred['DecisionTree']),
                                'RMSE': mean_squared_error(y_true['DecisionTree'], y_pred['DecisionTree'], squared=False),
                                'MAE': mean_absolute_error(y_true['DecisionTree'], y_pred['DecisionTree']),
                                'MSE': mean_squared_error(y_true['DecisionTree'], y_pred['DecisionTree'])
                            },
                            'RandomForest': {
                                'R2': r2_score(y_true['RandomForest'], y_pred['RandomForest']),
                                'RMSE': mean_squared_error(y_true['RandomForest'], y_pred['RandomForest'], squared=False),
                                'MAE': mean_absolute_error(y_true['RandomForest'], y_pred['RandomForest']),
                                'MSE': mean_squared_error(y_true['RandomForest'], y_pred['RandomForest'])
                            },
                            'XGBoost': {
                                'R2': r2_score(y_true['XGBoost'], y_pred['XGBoost']),
                                'RMSE': mean_squared_error(y_true['XGBoost'], y_pred['XGBoost'], squared=False),
                                'MAE': mean_absolute_error(y_true['XGBoost'], y_pred['XGBoost']),
                                'MSE': mean_squared_error(y_true['XGBoost'], y_pred['XGBoost'])
                            }
                        }

                        # Almacenar la tabla, el gráfico y las métricas en una lista
                        data.append({'sku': sku, 'table_html': table_html, 'graph': uri, 'metrics': metrics})

            except Exception as e:
                error_message = f"Ocurrió un error al procesar el archivo CSV: {str(e)}"

    return render(request, 'pronostic/dashboard.html', {'data': data, 'error_message': error_message})
