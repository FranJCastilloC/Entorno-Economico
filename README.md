# Navega a tu repositorio local
cd ruta/a/tu/repositorio

# Crea el archivo requirements.txt
echo "dash==2.14.0
pandas==2.1.1
numpy==1.25.2
scikit-learn==1.3.0
plotly==5.17.0
joblib==1.3.2
gunicorn==21.2.0" > requirements.txt

# Añade, haz commit y sube
git add requirements.txt
git commit -m "Add requirements.txt"
git push
