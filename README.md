# Обнаружение аномального потребления рекомендаций на данных Yambda

Данные: [Yambda-50M](https://huggingface.co/datasets/yandex/yambda)

## План

1. Разведочный анализ данных ✅ (`notebooks/01_eda.ipynb`)
2. Поиск кандидатов в аномалии ✅ (`notebooks/02_candidates.ipynb`, `src/bot_hunt.py`)
3. Признаки пользователей
4. Генерация синтетических аномалий
5. Детекторы: робастный z-score, [Isolation Forest](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest), автоэнкодер
6. Сравнение детекторов на синтетических данных
7. Проверка на реальных данных

## Литература

- [NIST: Detection of Outliers](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)
- [Liu et al. Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
