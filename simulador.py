import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
import pandas as pd
import seaborn as sns

# --------------------- PARÁMETROS CONFIGURABLES ---------------------
PARAMS = {
    "num_empresas_iniciales": 5,
    "num_clientes": 100,
    "num_ciclos": 26,
    "coste_fijo": 1500,
    "precio_coche": 2000,
    "prob_nueva_empresa": 0.05,
    "coches_fabricados_min": 5,
    "coches_fabricados_max": 10,
    "dinero_inicial_min": 10000,
    "dinero_inicial_max": 30000,
    "clientes_iniciales_min": 10,
    "clientes_iniciales_max": 30,
    "prob_base_compra": 0.05,
    "prob_fidelidad_base": 0.6,
    "prob_lanzar_campaña": 0.05,
    "max_duracion_campaña": 4,
    "tipos_campaña": {
        "descuento": {"prob_compra_bonus": 0.15, "coste": 800},
        "publicidad": {"prob_compra_bonus": 0.10, "coste": 600},
        "servicio": {"prob_fidelidad_bonus": 0.05, "coste": 400},
    },
    # Parámetros de financiación:
    "umbral_financiacion": 5000,
    "prob_financiacion": 0.5,
    "monto_financiacion": 10000,
    "interes_financiacion": 0.1,
    "pago_financiacion": 2000,
    "umbral_deuda": 20000,
    # Parámetros de fusión entre empresas:
    "umbral_fusion_pobre": 2000,
    "umbral_fusion_rica": 25000,
}

# Lista de proveedores con precio unitario y calidad.
PROVEEDORES = [
    {"nombre": "ProveedorA", "precio_unitario": 450, "calidad": 0.8},
    {"nombre": "ProveedorB", "precio_unitario": 480, "calidad": 0.9},
    {"nombre": "ProveedorC", "precio_unitario": 500, "calidad": 1.0},
    {"nombre": "ProveedorD", "precio_unitario": 430, "calidad": 0.7},
]

# --------------------- CONFIGURACIÓN DE SIMULACIONES ---------------------
NUM_SIMULACIONES = 60  # Número de ejecuciones
used_seeds = set()      # Para garantizar que cada semilla sea única
csv_filename = "resultados_simulaciones.csv"

with open(csv_filename, mode='w', newline='', encoding='utf-8') as fichero_csv:
    # Se añaden las dos nuevas columnas "simulacion" y "semilla" junto a las originales.
    campos = ["simulacion", "semilla", "ciclo", "empresa", "dinero", "clientes", "ventas", "participacion_mercado", "campana_activa", "proveedor"]
    escritor = csv.DictWriter(fichero_csv, fieldnames=campos)
    escritor.writeheader()

    # Bucle de simulaciones
    for sim in range(1, NUM_SIMULACIONES + 1):
        # --- Generación de semilla única ---
        semilla = random.randint(0, 10**9)
        while semilla in used_seeds:
            semilla = random.randint(0, 10**9)
        used_seeds.add(semilla)
        random.seed(semilla)
        print(f"Simulación {sim} usando semilla: {semilla}")

        # ------------------ CREACIÓN DE CLIENTES ------------------
        clientes_disponibles = [f"cliente{i+1}" for i in range(PARAMS["num_clientes"])]
        random.shuffle(clientes_disponibles)

        # ------------------ INICIALIZACIÓN DE EMPRESAS ------------------
        empresas = {}
        for i in range(PARAMS["num_empresas_iniciales"]):
            nombre = f"empresa{i+1}"
            num_clientes_inicial = random.randint(PARAMS["clientes_iniciales_min"], PARAMS["clientes_iniciales_max"])
            clientes_iniciales = [clientes_disponibles.pop() for _ in range(min(num_clientes_inicial, len(clientes_disponibles)))]
            dinero_inicial = random.randint(PARAMS["dinero_inicial_min"], PARAMS["dinero_inicial_max"])
            proveedor = random.choice(PROVEEDORES)
            estrategia = random.choice(["calidad", "economico"])
            empresas[nombre] = {
                "dinero": dinero_inicial,
                "clientes": clientes_iniciales,
                "hist_dinero": [dinero_inicial],
                "hist_clientes": [len(clientes_iniciales)],
                "hist_campanas": [],
                "hist_campana_nueva_duracion": [],
                "hist_proveedor": [proveedor["nombre"]],
                "hist_ventas": [],
                "hist_part_mercado": [],
                "hist_financiacion": [],
                "quebrada": False,
                "ciclo_quiebra": None,
                "coches_fabricados": random.randint(PARAMS["coches_fabricados_min"], PARAMS["coches_fabricados_max"]),
                "campana_activa": None,
                "proveedor": proveedor,
                "estrategia": estrategia,
                "deuda": 0
            }

        # Variable para registrar, por ciclo, el número de fusiones ocurridas.
        fusion_events_per_cycle = []

        # ------------------ BUCLE PRINCIPAL DE LA SIMULACIÓN (por ciclos) ------------------
        for ciclo in range(PARAMS["num_ciclos"]):
            
            # ------------------ GESTIÓN DE CAMPAÑAS Y CAMBIOS DE PROVEEDOR ------------------
            for nombre, datos in empresas.items():
                if datos["quebrada"]:
                    datos["hist_campanas"].append(None)
                    datos["hist_campana_nueva_duracion"].append(None)
                    datos["hist_proveedor"].append(None)
                    datos["hist_ventas"].append(0)
                    continue

                # Si hay campaña activa, se actualiza la duración.
                if datos["campana_activa"]:
                    datos["campana_activa"]["duracion"] -= 1
                    if datos["campana_activa"]["duracion"] <= 0:
                        datos["campana_activa"] = None

                # DECISIÓN ESTRATÉGICA PARA LANZAR CAMPAÑAS:
                new_campaign_duration = None  # Duración de la campaña si se lanza en este ciclo.
                if not datos["campana_activa"]:
                    campaign_max_cost = max(tipo["coste"] for tipo in PARAMS["tipos_campaña"].values())
                    dinero_minimo = PARAMS["coste_fijo"] + campaign_max_cost
                    puede_lanzar = False

                    if datos["dinero"] >= dinero_minimo:
                        if ciclo > 0 and datos["hist_ventas"]:
                            ventas_anteriores = datos["hist_ventas"][-1]
                            if ventas_anteriores < 0.5 * datos["coches_fabricados"]:
                                puede_lanzar = True
                        else:
                            puede_lanzar = True

                    if puede_lanzar and random.random() < PARAMS["prob_lanzar_campaña"]:
                        tipo = random.choice(list(PARAMS["tipos_campaña"].keys()))
                        duracion = random.randint(1, PARAMS["max_duracion_campaña"])
                        datos["campana_activa"] = {"tipo": tipo, "duracion": duracion}
                        new_campaign_duration = duracion

                # Registro del tipo de campaña y, de ser nueva, su duración.
                datos["hist_campanas"].append(datos["campana_activa"]["tipo"] if datos["campana_activa"] else None)
                datos["hist_campana_nueva_duracion"].append(new_campaign_duration)

                # CAMBIO DE PROVEEDOR SEGÚN ESTRATEGIA:
                if random.random() < 0.05:
                    nuevos = [p for p in PROVEEDORES if p != datos["proveedor"]]
                    if datos["estrategia"] == "calidad":
                        datos["proveedor"] = max(nuevos, key=lambda p: p["calidad"])
                    else:
                        datos["proveedor"] = min(nuevos, key=lambda p: p["precio_unitario"])
                datos["hist_proveedor"].append(datos["proveedor"]["nombre"])
            
            # ------------------ COMPORTAMIENTO DE LOS CLIENTES ------------------
            for cliente in [f"cliente{i+1}" for i in range(PARAMS["num_clientes"])]:
                actual = next((n for n, d in empresas.items() if cliente in d["clientes"]), None)
                if actual and not empresas[actual]["quebrada"]:
                    campana = empresas[actual]["campana_activa"]
                    prob = PARAMS["prob_base_compra"]
                    if campana and "prob_compra_bonus" in PARAMS["tipos_campaña"][campana["tipo"]]:
                        prob += PARAMS["tipos_campaña"][campana["tipo"]]["prob_compra_bonus"]
                    if random.random() < prob:
                        continue
                    else:
                        fidelidad = PARAMS["prob_fidelidad_base"]
                        if campana and "prob_fidelidad_bonus" in PARAMS["tipos_campaña"][campana["tipo"]]:
                            fidelidad += PARAMS["tipos_campaña"][campana["tipo"]]["prob_fidelidad_bonus"]
                        if random.random() > fidelidad:
                            empresas[actual]["clientes"].remove(cliente)
                            candidatas = [n for n, d in empresas.items() if not d["quebrada"] and n != actual]
                            if candidatas:
                                nueva = random.choice(candidatas)
                                empresas[nueva]["clientes"].append(cliente)
            
            total_ventas_ciclo = 0
            ventas_por_empresa = {}
            # ------------------ GESTIÓN FINANCIERA, PRODUCCIÓN, VENTAS Y FINANCIACIÓN ------------------
            for nombre, datos in empresas.items():
                if datos["quebrada"]:
                    datos["hist_dinero"].append(datos["dinero"])
                    datos["hist_clientes"].append(0)
                    datos["hist_ventas"].append(0)
                    datos["hist_part_mercado"].append(0)
                    continue

                proveedor = datos["proveedor"]
                coste_fabricacion = datos["coches_fabricados"] * proveedor["precio_unitario"]
                coste_total = coste_fabricacion + PARAMS["coste_fijo"]
                if datos["campana_activa"]:
                    tipo = datos["campana_activa"]["tipo"]
                    coste_total += PARAMS["tipos_campaña"][tipo]["coste"]

                vendidos = random.randint(0, datos["coches_fabricados"])
                ingresos = vendidos * PARAMS["precio_coche"]
                datos["dinero"] += ingresos - coste_total

                # ------------------ BLOQUE DE FINANCIACIÓN ------------------
                financiacion_event = 0
                if datos["deuda"] > 0:
                    datos["deuda"] *= (1 + PARAMS["interes_financiacion"])
                    pago = min(PARAMS["pago_financiacion"], datos["dinero"], datos["deuda"])
                    datos["dinero"] -= pago
                    datos["deuda"] -= pago
                else:
                    if datos["dinero"] < PARAMS["umbral_financiacion"]:
                        if random.random() < PARAMS["prob_financiacion"]:
                            financiacion_event = PARAMS["monto_financiacion"]
                            datos["dinero"] += financiacion_event
                            datos["deuda"] = financiacion_event
                datos["hist_financiacion"].append(financiacion_event)

                # Si la deuda supera el umbral o si el dinero queda negativo, se declara quiebra.
                if datos["deuda"] >= PARAMS["umbral_deuda"]:
                    datos["quebrada"] = True
                    datos["dinero"] = 0
                    datos["ciclo_quiebra"] = ciclo
                if datos["dinero"] < 0:
                    datos["quebrada"] = True
                    datos["dinero"] = 0
                    datos["ciclo_quiebra"] = ciclo

                datos["hist_dinero"].append(datos["dinero"])
                datos["hist_clientes"].append(len(datos["clientes"]))
                datos["hist_ventas"].append(vendidos)
                ventas_por_empresa[nombre] = vendidos
                total_ventas_ciclo += vendidos

            for nombre, datos in empresas.items():
                ventas = ventas_por_empresa.get(nombre, 0)
                datos["hist_part_mercado"].append(100 * ventas / total_ventas_ciclo if total_ventas_ciclo > 0 else 0)

            # ------------------ FUSIÓN ENTRE EMPRESAS ------------------
            fusion_count = 0
            current_companies = list(empresas.keys())
            for poor_company in current_companies:
                if poor_company in empresas and empresas[poor_company]["dinero"] < PARAMS["umbral_fusion_pobre"]:
                    rich_candidates = [comp for comp in empresas if comp != poor_company and empresas[comp]["dinero"] > PARAMS["umbral_fusion_rica"]]
                    if rich_candidates:
                        rich_company = random.choice(rich_candidates)
                        rich = empresas[rich_company]
                        poor = empresas[poor_company]
                        rich["dinero"] += poor["dinero"]
                        rich["clientes"].extend(poor["clientes"])
                        rich["coches_fabricados"] += poor["coches_fabricados"]
                        rich["deuda"] += poor["deuda"]
                        fusion_count += 1
                        print(f"Fusión en ciclo {ciclo}: {rich_company} adquirió a {poor_company}")
                        del empresas[poor_company]
            fusion_events_per_cycle.append(fusion_count)
        # Fin del bucle de ciclos

        # ------------------ GUARDAR RESULTADOS EN CSV ------------------
        # Se recorren los ciclos (incluyendo el ciclo 0 con la condición inicial)
        for ciclo in range(PARAMS["num_ciclos"] + 1):
            for nombre, datos in empresas.items():
                try:
                    dinero = datos["hist_dinero"][ciclo]
                    clientes_count = datos["hist_clientes"][ciclo]
                    ventas = datos["hist_ventas"][ciclo]
                    participacion = datos["hist_part_mercado"][ciclo]
                    campana = datos["hist_campanas"][ciclo] if ciclo < len(datos["hist_campanas"]) else None
                    proveedor_nombre = datos["hist_proveedor"][ciclo] if ciclo < len(datos["hist_proveedor"]) else None
                except IndexError:
                    dinero = clientes_count = ventas = participacion = campana = proveedor_nombre = None
                escritor.writerow({
                    "simulacion": sim,
                    "semilla": semilla,
                    "ciclo": ciclo,
                    "empresa": nombre,
                    "dinero": dinero,
                    "clientes": clientes_count,
                    "ventas": ventas,
                    "participacion_mercado": participacion,
                    "campana_activa": campana,
                    "proveedor": proveedor_nombre
                })

        # ------------------ VISUALIZACIÓN (OPCIONAL) ------------------
        # Mostramos los gráficos únicamente en la última simulación para no abrir 60 ventanas.
        if sim == NUM_SIMULACIONES:
            # Visualización Principal (4 gráficos)
            fig, axs = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
            axs[-1].set_xlabel("Número de ciclo")
            for nombre, datos in empresas.items():
                axs[0].plot(datos["hist_dinero"], label=nombre)
                if datos["quebrada"]:
                    axs[0].axvline(x=datos["ciclo_quiebra"], color='red', linestyle='--', alpha=0.6)
            axs[0].set_title("Dinero de las empresas a lo largo del tiempo")
            axs[0].set_ylabel("Dinero (€)")
            axs[0].legend()
            axs[0].grid(True)
            for nombre, datos in empresas.items():
                axs[1].plot(datos["hist_clientes"], label=nombre)
            axs[1].set_title("Número de clientes de las empresas a lo largo del tiempo")
            axs[1].set_ylabel("Clientes")
            axs[1].legend()
            axs[1].grid(True)
            for nombre, datos in empresas.items():
                axs[2].plot(datos["hist_ventas"], label=nombre)
            axs[2].set_title("Ventas de coches por empresa en cada ciclo")
            axs[2].set_ylabel("Coches vendidos")
            axs[2].legend()
            axs[2].grid(True)
            for nombre, datos in empresas.items():
                axs[3].plot(datos["hist_part_mercado"], label=nombre)
            axs[3].set_title("Participación de mercado (%) por empresa en cada ciclo")
            axs[3].set_ylabel("Participación (%)")
            axs[3].legend()
            axs[3].grid(True)
            plt.tight_layout()
            plt.show()

            # Gráficos Extra Unificados (2 gráficos)
            # Datos para campañas:
            campaign_counts = []
            campaign_avg_duration = []
            for cyc in range(PARAMS["num_ciclos"] + 1):
                count = 0
                duration_sum = 0
                for empresa, datos in empresas.items():
                    if cyc < len(datos["hist_campana_nueva_duracion"]) and datos["hist_campana_nueva_duracion"][cyc] is not None:
                        count += 1
                        duration_sum += datos["hist_campana_nueva_duracion"][cyc]
                campaign_counts.append(count)
                campaign_avg_duration.append(duration_sum / count if count > 0 else 0)
            # Datos para financiaciones:
            financing_counts = []
            financing_total = []
            for cyc in range(PARAMS["num_ciclos"] + 1):
                count = 0
                total_amount = 0
                for empresa, datos in empresas.items():
                    if cyc < len(datos["hist_financiacion"]):
                        amount = datos["hist_financiacion"][cyc]
                        if amount > 0:
                            count += 1
                            total_amount += amount
                financing_counts.append(count)
                financing_total.append(total_amount)
            fig_extra, (ax_extra1, ax_extra2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
            ax_extra1.bar(range(PARAMS["num_ciclos"] + 1), campaign_counts, color='skyblue', label="Campañas lanzadas")
            ax_extra1.set_ylabel("Número de campañas", color='blue')
            ax_extra1.tick_params(axis='y', labelcolor='blue')
            ax_extra1.set_title("Campañas realizadas por ciclo y duración promedio")
            ax1_twin = ax_extra1.twinx()
            ax1_twin.plot(range(PARAMS["num_ciclos"] + 1), campaign_avg_duration, color='red', marker='o', label="Duración promedio")
            ax1_twin.set_ylabel("Duración (ciclos)", color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            l1, lab1 = ax_extra1.get_legend_handles_labels()
            l2, lab2 = ax1_twin.get_legend_handles_labels()
            ax_extra1.legend(l1+l2, lab1+lab2, loc='upper right')

            ax_extra2.bar(range(PARAMS["num_ciclos"] + 1), financing_counts, color='lightgreen', label="Eventos de financiación")
            ax_extra2.set_xlabel("Ciclo")
            ax_extra2.set_ylabel("Número de financiaciones", color='green')
            ax_extra2.tick_params(axis='y', labelcolor='green')
            ax_extra2.set_title("Financiaciones realizadas por ciclo")
            ax2_twin = ax_extra2.twinx()
            ax2_twin.plot(range(PARAMS["num_ciclos"] + 1), financing_total, color='purple', marker='o', label="Monto financiado")
            ax2_twin.set_ylabel("Monto financiado (€)", color='purple')
            ax2_twin.tick_params(axis='y', labelcolor='purple')
            l3, lab3 = ax_extra2.get_legend_handles_labels()
            l4, lab4 = ax2_twin.get_legend_handles_labels()
            ax_extra2.legend(l3+l4, lab3+lab4, loc='upper right')
            plt.tight_layout()
            plt.show()

print(f"Datos guardados en '{csv_filename}'")

def generar_tabla_visual(csv_filename):
    import pandas as pd
    # Leer el CSV
    df = pd.read_csv(csv_filename)
    # Ordenar el DataFrame y reiniciar el índice para no mostrar el índice original
    df = df.sort_values(by=["simulacion", "ciclo", "empresa"]).reset_index(drop=True)
    # Convertir el DataFrame a HTML con un estilo (aquí se aplican clases de Bootstrap, por ejemplo)
    html_table = df.style.set_table_attributes('class="table table-bordered table-striped"').to_html()
    # Guardar la tabla en un archivo HTML
    with open("tabla_resultados.html", "w", encoding="utf-8") as f:
        f.write(html_table)
    print("Tabla de resultados guardada en 'tabla_resultados.html'")


# Llamamos a la función
generar_tabla_visual(csv_filename)

def calcular_estadisticas(csv_filename):
    # Leer el CSV generado por el simulador
    df = pd.read_csv(csv_filename)

    # Ordenar los ciclos y obtener el penúltimo ciclo
    ciclos_ordenados = sorted(df["ciclo"].unique(), reverse=True)
    if len(ciclos_ordenados) < 2:
        print("No hay suficientes ciclos para calcular el penúltimo.")
        return None, None  # Retorno seguro para evitar errores
    
    penultimo_ciclo = ciclos_ordenados[1]  # Segundo más grande
    df_final = df[df["ciclo"] == penultimo_ciclo]  # Filtrar datos del penúltimo ciclo
    
    # Agrupar los datos por simulación y calcular estadísticas clave
    stats = df_final.groupby("simulacion").agg({
        "dinero": ["mean", "min", "max", "std"],
        "clientes": ["mean", "min", "max", "std"],
        "ventas": ["mean", "min", "max", "std"],
        "participacion_mercado": ["mean", "min", "max", "std"]
    })

    print(f"Estadísticas finales por simulación en el ciclo {penultimo_ciclo}:")
    print(stats)

    # Calcular estadísticas globales en el penúltimo ciclo
    global_stats = df_final[['dinero', 'clientes', 'ventas', 'participacion_mercado']].agg(["mean", "min", "max", "std"])
    print("\nEstadísticas globales:")
    print(global_stats)

    # Mostrar evolución del dinero, clientes, ventas y participacion del mercado promedio a lo largo de los ciclos
    df_grouped = df.groupby("ciclo")[["dinero", "clientes", "ventas", "participacion_mercado"]].mean()

    #df_grouped = df.groupby("ciclo")["dinero"].mean()
    #print("\nEvolución del Dinero Promedio a lo largo de los Ciclos:")
    #print(df_grouped)

    return df, df_final  # Retorna ambos DataFrames para usarlos en los gráficos

# Llamar a la función y obtener los datos
df, df_final = calcular_estadisticas(csv_filename)

if df is not None and df_final is not None:
    
    def grafico_evolucion(df):
        # Agrupar por ciclo y calcular el promedio de las variables clave
        df_grouped = df.groupby("ciclo")[["dinero", "clientes", "ventas", "participacion_mercado"]].mean()

        # Crear figura y dos ejes y
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # EJE PRINCIPAL: Dinero (valores grandes)
        ax1.plot(df_grouped.index, df_grouped["dinero"], marker="o", color="blue", label="dinero")
        ax1.set_xlabel("Ciclo")
        ax1.set_ylabel("Dinero (€)", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")

        # EJE SECUNDARIO: Clientes, Ventas y Participación de mercado
        ax2 = ax1.twinx()
        ax2.plot(df_grouped.index, df_grouped["clientes"], marker="o", color="orange", label="clientes")
        ax2.plot(df_grouped.index, df_grouped["ventas"], marker="o", color="green", label="ventas")
        ax2.plot(df_grouped.index, df_grouped["participacion_mercado"], marker="o", color="red", label="participacion_mercado")
        ax2.set_ylabel("Clientes / Ventas / Participación (%)", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        # Combinar leyendas de ambos ejes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Título y formato
        plt.title("Evolución de las variables a lo largo de los ciclos")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def grafico_distribucion(df_final):
        plt.figure(figsize=(12, 6))
        for i, columna in enumerate(["dinero", "clientes", "ventas", "participacion_mercado"], 1):
            plt.subplot(2, 2, i)
            sns.histplot(df_final[columna], kde=True)
            plt.title(f"Distribución de {columna}")

        plt.tight_layout()
        plt.show()

    def grafico_comparacion(df_final):
        # Agrupamos por simulación y calculamos los promedios
        stats_grouped = df_final.groupby("simulacion")[["dinero", "clientes", "ventas", "participacion_mercado"]].mean()

        # Crear subgráficos 2x2
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        # Gráfico de dinero
        stats_grouped["dinero"].plot(kind="bar", ax=axs[0, 0], color="blue")
        axs[0, 0].set_title("Dinero promedio por simulación")
        axs[0, 0].set_ylabel("€")
        axs[0, 0].grid(True)

        # Gráfico de clientes
        stats_grouped["clientes"].plot(kind="bar", ax=axs[0, 1], color="orange")
        axs[0, 1].set_title("Clientes promedio por simulación")
        axs[0, 1].set_ylabel("Clientes")
        axs[0, 1].grid(True)

        # Gráfico de ventas
        stats_grouped["ventas"].plot(kind="bar", ax=axs[1, 0], color="green")
        axs[1, 0].set_title("Ventas promedio por simulación")
        axs[1, 0].set_ylabel("Coches vendidos")
        axs[1, 0].grid(True)

        # Gráfico de participación de mercado
        stats_grouped["participacion_mercado"].plot(kind="bar", ax=axs[1, 1], color="red")
        axs[1, 1].set_title("Participación de mercado promedio (%)")
        axs[1, 1].set_ylabel("%")
        axs[1, 1].grid(True)

        # Ajustes finales
        for ax in axs.flat:
            ax.set_xlabel("Simulación")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.suptitle("Comparación entre simulaciones por variable")
        plt.tight_layout()
        plt.show()


    def grafico_correlacion(df_final):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_final["dinero"], y=df_final["participacion_mercado"])
        plt.xlabel("Dinero (€)")
        plt.ylabel("Participación de mercado")
        plt.title("Relación entre Dinero y Participación de Mercado")
        plt.grid(True)
        plt.show()

    # Generar los gráficos con los datos correctos
    grafico_evolucion(df)
    grafico_distribucion(df_final)
    grafico_comparacion(df_final)
    grafico_correlacion(df_final)
