from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('carga_csv/', views.carga_csv, name='carga_csv'),
    path('logout/', views.logout_view, name='logout'),  
    path('graficos/', views.graficos, name='graficos'),
    path('decision_tree/', views.decision_tree, name='decision_tree'),
    path('graficos_tree/', views.graficos_tree, name='graficos_tree'),
    path('profeta/', views.profeta_vista, name='profeta_vista'),
    path('profeta_tabla/', views.profeta_tabla, name='profeta_tabla'),
    path('random_forest/', views.random_forest, name='random_forest'),
    path('xgboost/', views.xgboost, name='xgboost'),
    path('dashboard/', views.dashboard, name='dashboard')
    
]
