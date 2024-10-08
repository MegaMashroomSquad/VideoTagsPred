Вас приветствует команда Prod Legit!

В репозитории представлено решение кейса предсказания тегов видео.
Кейсодержатель: Rutube
Всероссийский хакатон Цифровой Прорыв

Состав команды:
Фролов Александр (2 курс, Бизнес-информатика, ВШЭ)
Куценко Дмитрий (2 курс, ПМИ, ВШЭ)
Степанов Юрий (1 курс, ПМИ, ВШЭ)
Васильев Виктор (1 курс, ПМИ, ВШЭ)
Серегин Владислав (1 курс, ПМИ, ВШЭ)

Краткое описание нашего решения:
каждое видео из датасета преобразуем в текст (из аудио дорожки)
применяем суммаризацию для каждого такого текста из видео, а также для названия и описания
обучаем LLM модель на целевую фичу “tags”
деплоим модель на стримлит для удобства представления

Дальше будет описан pipeline наших действий:
транскрибируем видео (конкретно аудио) благодаря whisper в текст
проводим чистку текста от стоп-слов и лишних символов (пунктуация, числа и т.д.)
суммаризация (abstractive ru_t5)
балансировка классов
data_split на train и validation
обучение модели (lighautoml, или vikhr из NeMo, или catboost-на нем финальное решение) с целевой переменной tags, которая предсказывает сначала более общие теги, а затем более специфические (по иерархии)
оценка качества (IoU метрика) + кросс-валидация
тестирование для лидер борда
деплой модели на streamlit для показа на защите

Дополнительно рассматриваемые варианты решений:
после суммаризации с помощью ner, или spacy, или natasha вытащить из текста именованные сущности (теги), преобразовать их в векторное представление и сопоставить их ближайшим нужным тегам
на примере существующих сэмплов суммаризации генерируем смежные синтетические данные для большего количества представителей классов
использование иерархической кластеризации для группировки тегов разных уровней в один для поиска наиболее релевантных
соотносится с предыдущим: объединить теги на большие смысловые группы для решения проблемы малого членства в датасете
к суммаризации транскрибированного аудио, описания и названия добавить выявление признаков из видеоряда
иерархия тегов может быть представлена в виде графа, где каждый тег — это узел, а связи между ними определяют иерархические отношения

Преимущества нашего решения:
мы используем мощную модель whisper по преобразованию аудио в текст, ведь качественная транскрипция позволяет максимально точно преобразовать аудиоконтент в текст, что критически важно для последующей работы с LLM
80 Гб видео будут долго обучаться без предобработки, поэтому мы используем “зачистку” текста от не несущих смысл слов-токенов, а также суммаризацию, сохраняя все ключевые моменты контекста видео
использование предобученной модели catboost - скорость и гибкость
предсказание тегов поэтапно/поуровнево - сначала обобщенные и простые теги (1 уровень иерархии), потом углубление в детали (2 и 3 уровни)
балансировка классов (+генерирование синтетических данных) - борьба с малым количеством представителей классов (чтобы не просто их игнорировать, а также точно уметь работать)
использования кросс-валидации позволяет избежать переобучения
удобство и наглядность представляемого результата благодаря стримлиту


Как наше решение поможет видеохостингу?
улучшение поиска и навигации (по фильтрам и ключевым словам) благодаря иерархической категоризации контента - пользователю легко найти нужный материал => он будет чаще обращаться к сервису
повышение качества рекомендательной системы (пользователи будут довольны предлагаемыми им видео и будут проводить на хостинге больше времени)
экономия времени создателей контента и минимизация “вкусового фактора”: например, для одних это видео про машины, для других про технологии - ml модель будет создавать свои единые стандарты, а также предоставлять сразу список тегов, а не только те, которые придумал пользователь
легкость в масштабировании (например, мультиязычная поддержка - теги доступны для контента на разных языках)
точные теги помогают с фильтрацией и модерацией контента
подсчет статистики, что позволяет авторам лучше понимать предпочтения зрителей
Как итог: данное решение улучшит взаимодействие как пользователей, так и создателей контента, повышая качество сервиса Rutube.

0.62 IoU лежит в ветке it_shnik
