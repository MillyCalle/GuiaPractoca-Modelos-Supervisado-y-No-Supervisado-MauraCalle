import os
import glob
import pandas as pd
import win32com.client as win32

# Carpeta donde están los archivos .xls
ruta = "data"

# Buscar .xls
archivos_xls = glob.glob(os.path.join(ruta, "*.xls"))

print("Archivos encontrados:")
for a in archivos_xls:
    print(" -", a)

# Carpeta para convertir xlsx
carpeta_xlsx = os.path.join(ruta, "convertidos")
os.makedirs(carpeta_xlsx, exist_ok=True)

print("\nConvirtiendo XLS → XLSX...")

excel = win32.Dispatch("Excel.Application")
excel.Visible = False

archivos_xlsx = []

for xls in archivos_xls:
    nombre = os.path.basename(xls).replace(".xls", ".xlsx")
    destino = os.path.join(carpeta_xlsx, nombre)

    wb = excel.Workbooks.Open(os.path.abspath(xls))
    wb.SaveAs(os.path.abspath(destino), FileFormat=51)  # 51 = xlsx
    wb.Close()

    archivos_xlsx.append(destino)
    print("Convertido:", destino)

excel.Quit()

print("\nLeyendo archivos XLSX convertidos...")

dfs = []
for xlsx in archivos_xlsx:
    df_temp = pd.read_excel(xlsx, engine="openpyxl")
    dfs.append(df_temp)

print("\nConcatenando todo...")

master = pd.concat(dfs, ignore_index=True)

salida = os.path.join(ruta, "academic_performance_master.csv")
master.to_csv(salida, index=False)

print("\n✔️ Archivo final generado con éxito:")
print("   →", salida)
