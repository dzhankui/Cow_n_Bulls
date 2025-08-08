import streamlit as st
import pandas as pd
import csv
import heapq
from collections import defaultdict
import time

st.set_page_config(page_title="Оптимизатор селекции КРС", page_icon="🐄", layout="wide")


# Функция для вычисления коэффициента родства
def kinship(id1, id2, memo, pedigree):
	if id1 is None or id2 is None:
		return 0.0
	if id1 == id2:
		mother, father = pedigree.get(id1, (None, None))
		F = kinship(father, mother, memo, pedigree) if mother and father else 0.0
		return 0.5 * (1 + F)
	
	if id1 > id2:
		id1, id2 = id2, id1
	key = (id1, id2)
	if key in memo:
		return memo[key]
	
	mother2, father2 = pedigree.get(id2, (None, None))
	a_val = 0.0
	count = 0
	if father2:
		a_val += kinship(id1, father2, memo, pedigree)
		count += 1
	if mother2:
		a_val += kinship(id1, mother2, memo, pedigree)
		count += 1
	
	a_val = a_val * 0.5 if count else 0.0
	memo[key] = a_val
	return a_val


# Основная функция обработки данных
def process_data(pedigree_file, bulls_file, cows_file, max_kinship, max_usage_percent):
	# Загрузка данных
	pedigree_df = pd.read_csv(pedigree_file)
	bulls_df = pd.read_csv(bulls_file)
	cows_df = pd.read_csv(cows_file)
	
	# Преобразование в словари и списки
	pedigree = {}
	for _, row in pedigree_df.iterrows():
		id_ = str(row['id'])
		mother = str(row['mother_id']) if pd.notna(row['mother_id']) else None
		father = str(row['father_id']) if pd.notna(row['father_id']) else None
		pedigree[id_] = (mother, father)
	
	bulls = []
	for _, row in bulls_df.iterrows():
		bulls.append({'id': str(row['id']), 'ebv': float(row['ebv'])})
	
	cows = []
	for _, row in cows_df.iterrows():
		cows.append({'id': str(row['id']), 'ebv': float(row['ebv'])})
	
	# Добавление отсутствующих животных в родословную
	all_animals = set(pedigree.keys()) | {b['id'] for b in bulls} | {c['id'] for c in cows}
	for animal in all_animals:
		if animal not in pedigree:
			pedigree[animal] = (None, None)
	
	# Вычисление допустимых пар
	memo = {}
	bull_ebv = {b['id']: b['ebv'] for b in bulls}
	n_cows = len(cows)
	
	# Динамический расчет максимального числа использований
	max_uses = max(1, int(n_cows * max_usage_percent / 100))
	
	# Прогресс-бар
	progress_bar = st.progress(0)
	status_text = st.empty()
	
	# Построение списка кандидатов
	cow_candidates = []
	valid_pairs_count = 0
	rejected_pairs = []
	
	for i, cow in enumerate(cows):
		candidates = []
		for bull in bulls:
			a_val = kinship(bull['id'], cow['id'], memo, pedigree)
			if a_val <= max_kinship:
				combined_sq = (bull['ebv'] + cow['ebv']) ** 2
				candidates.append((bull['id'], bull['ebv'], combined_sq))
				valid_pairs_count += 1
			else:
				rejected_pairs.append({
					'bull_id': bull['id'],
					'bull_ebv': bull['ebv'],
					'cow_id': cow['id'],
					'cow_ebv': cow['ebv'],
					'kinship': a_val
				})
		candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
		cow_candidates.append([c[0] for c in candidates])
		
		# Обновление прогресса
		progress = (i + 1) / n_cows * 0.5
		progress_bar.progress(progress)
		status_text.text(f"Обработано коров: {i + 1}/{n_cows} | Найдено пар: {valid_pairs_count} | Отклонено: {len(rejected_pairs)}")
	
	# Назначение быков
	assignment = [None] * n_cows
	bull_use_count = {bull['id']: 0 for bull in bulls}
	current_index = [0] * n_cows
	heap = []
	assigned_count = 0
	
	for i in range(n_cows):
		if cow_candidates[i]:
			bull_id = cow_candidates[i][0]
			priority = bull_ebv[bull_id]
			heapq.heappush(heap, (-priority, i, 0))
	
	while heap:
		neg_priority, cow_idx, idx = heapq.heappop(heap)
		if idx >= len(cow_candidates[cow_idx]):
			continue
		
		bull_id = cow_candidates[cow_idx][idx]
		if bull_use_count.get(bull_id, 0) < max_uses:
			assignment[cow_idx] = bull_id
			bull_use_count[bull_id] = bull_use_count.get(bull_id, 0) + 1
			assigned_count += 1
		else:
			next_idx = idx + 1
			if next_idx < len(cow_candidates[cow_idx]):
				current_index[cow_idx] = next_idx
				next_bull_id = cow_candidates[cow_idx][next_idx]
				next_priority = bull_ebv[next_bull_id]
				heapq.heappush(heap, (-next_priority, cow_idx, next_idx))
		
		# Обновление прогресса
		progress = 0.5 + (assigned_count / n_cows) * 0.5
		progress_bar.progress(min(progress, 1.0))
		status_text.text(f"Назначено пар: {assigned_count}/{n_cows} | Макс. использований: {max_uses} | Обработка...")
	
	# Создание результирующего DataFrame
	result_data = []
	for cow, bull_id in zip(cows, assignment):
		result_data.append({
			'cow_id': cow['id'],
			'cow_ebv': cow['ebv'],
			'assigned_bull_id': bull_id if bull_id else 'N/A',
			'bull_ebv': bull_ebv.get(bull_id, 'N/A'),
			'expected_ebv': (cow['ebv'] + bull_ebv[bull_id]) / 2 if bull_id else 'N/A'
		})
	
	result_df = pd.DataFrame(result_data)
	
	# Статистика
	stats = {
		"Всего коров": n_cows,
		"Успешно назначено": f"{assigned_count} ({assigned_count / n_cows * 100:.1f}%)",
		"Средний EBV потомства": result_df[result_df['expected_ebv'] != 'N/A']['expected_ebv'].mean(),
		"Макс. использований на быка": max_uses,
		"Не назначено": n_cows - assigned_count,
		"Макс. допустимое родство": f"{max_kinship * 100:.1f}%",
		"Макс. % использования быка": f"{max_usage_percent}%"
	}
	
	progress_bar.empty()
	status_text.empty()
	
	return result_df, stats, pd.DataFrame(rejected_pairs)


# Интерфейс Streamlit
st.title("🐄 Оптимизатор селекции крупного рогатого скота")
st.markdown("""
**Задача:** Максимизировать среднюю ожидаемую селекционную ценность потомства (EBV) при сохранении генетического разнообразия.

**Ограничения:**
- Родство в паре не превышает заданный порог
- Один бык не может осеменить более установленного процента коров
""")

# Настройки параметров
st.header("⚙️ Параметры оптимизации")
col1, col2 = st.columns(2)
with col1:
	max_kinship = st.slider(
		"Максимально допустимый коэффициент родства",
		min_value=0.0,
		max_value=0.5,
		value=0.05,
		step=0.01,
		format="%.2f",
		help="Максимально допустимая генетическая схожесть между быком и коровой"
	)
with col2:
	max_usage_percent = st.slider(
		"Максимальный процент использования одного быка",
		min_value=1,
		max_value=100,
		value=10,
		step=1,
		format="%d%%",
		help="Максимальный процент коров, которых может осеменить один бык"
	)

# Загрузка файлов
st.header("📂 Загрузка данных")
col1, col2, col3 = st.columns(3)
with col1:
	pedigree_file = st.file_uploader("Родословные (pedigree.csv)", type="csv")
with col2:
	bulls_file = st.file_uploader("Быки (bulls.csv)", type="csv")
with col3:
	cows_file = st.file_uploader("Коровы (cows.csv)", type="csv")

# Примеры файлов
with st.expander("Примеры входных файлов"):
	st.markdown("**pedigree.csv**")
	st.code("""id,mother_id,father_id
1,,
2,,
3,1,2
4,1,2
5,3,4""")
	
	st.markdown("**bulls.csv**")
	st.code("""id,ebv
B001,120
B002,115
B003,125""")
	
	st.markdown("**cows.csv**")
	st.code("""id,ebv
C001,100
C002,105
C003,95""")

# Обработка данных
if st.button("Запустить оптимизацию", disabled=(pedigree_file is None or bulls_file is None or cows_file is None)):
	if pedigree_file and bulls_file and cows_file:
		with st.spinner("Обработка данных..."):
			start_time = time.time()
			result_df, stats, rejected_df = process_data(
				pedigree_file,
				bulls_file,
				cows_file,
				max_kinship,
				max_usage_percent
			)
			processing_time = time.time() - start_time
		
		st.success(f"Оптимизация завершена за {processing_time:.2f} секунд!")
		
		# Отображение статистики
		st.header("📊 Результаты оптимизации")
		stats_df = pd.DataFrame(list(stats.items()), columns=["Параметр", "Значение"])
		st.table(stats_df)
		
		# Визуализация
		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Распределение ожидаемого EBV потомства")
			if 'expected_ebv' in result_df.columns:
				valid_ebv = result_df[result_df['expected_ebv'] != 'N/A']['expected_ebv']
				if not valid_ebv.empty:
					st.bar_chart(valid_ebv.value_counts().sort_index())
		
		with col2:
			st.subheader("Использование быков")
			if 'assigned_bull_id' in result_df.columns:
				bull_usage = result_df[result_df['assigned_bull_id'] != 'N/A']['assigned_bull_id'].value_counts()
				if not bull_usage.empty:
					st.bar_chart(bull_usage)
		
		# Отображение назначений
		st.header("🧬 Назначения быков коровам")
		st.dataframe(result_df.style.format({"expected_ebv": "{:.2f}"}), height=500)
		
		# Отображение отклоненных пар
		if not rejected_df.empty:
			with st.expander("Просмотр отклоненных пар (из-за родства)"):
				st.dataframe(rejected_df)
		
		# Экспорт результатов
		st.header("📤 Экспорт результатов")
		csv_data = result_df.to_csv(index=False).encode('utf-8')
		st.download_button(
			label="Скачать назначения (CSV)",
			data=csv_data,
			file_name="bull_assignments.csv",
			mime="text/csv"
		)
	else:
		st.warning("Пожалуйста, загрузите все файлы!")

# Информация о работе алгоритма
with st.expander("ℹ️ Как работает алгоритм"):
	st.markdown(f"""
**Алгоритм оптимизации с гибкими параметрами:**

1. **Пользовательские настройки:**
   - Макс. родство: `{max_kinship}` (фильтрация генетически близких пар)
   - Макс. использование быка: `{max_usage_percent}%` (контроль генетического разнообразия)

2. **Основные этапы:**
   - Расчет коэффициента родства для всех возможных пар
   - Фильтрация пар с родством > `{max_kinship}`
   - Формирование списка кандидатов для каждой коровы
   - Жадное назначение с учетом ограничения `{max_usage_percent}%`
   - Расчет ожидаемого EBV потомства

3. **Оптимизационные критерии:**
   - Максимизация среднего EBV потомства
   - Минимизация инбридинга
   - Равномерное использование быков-производителей
""")

st.markdown("---")
st.caption("© 2025 Оптимизатор селекции КРС | Разработано dzhankui@bk.ru для повышения эффективности животноводства")