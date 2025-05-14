db = db.getSiblingDB('reps-database');

db.createCollection('users', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'email', 'password'],
            properties: {
                name: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                email: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                password: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                }
            }
        }
    }
});

db.users.createIndex({ email: 1 }, { unique: true });

db.createCollection('exercises', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'muscle_group', 'short_description', 'long_description', 'image_path', 'original_video_path', 'landmarks_data_path'],
            properties: {
                name: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                muscle_group: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                short_description: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                long_description: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                image_path: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                original_video_path: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                },
                landmarks_data_path: {
                    bsonType: 'string',
                    description: 'Must be a string and is required'
                }
            }
        }
    }
});

db.exercises.createIndex({ name: 1 }, { unique: true });

// Inserting data
db.exercises.insertMany([
    {
        name: 'Remo con Barra',
        muscle_group: 'Espalda',
        short_description: 'Ejercicio de tracción que trabaja la espalda alta, dorsales y trapecios, junto con los bíceps, desarrollando fuerza y masa muscular en la espalda.',
        long_description: '<p>El remo con barra es un ejercicio de tracción horizontal que se centra en el fortalecimiento de la espalda alta, los dorsales, los trapecios y los bíceps. Es ideal para desarrollar masa muscular y mejorar la postura general.</p><p>Para realizarlo, el atleta sostiene una barra con ambas manos, inclinándose hacia adelante desde las caderas mientras mantiene la espalda recta. Luego, tira de la barra hacia el torso, apretando los omóplatos antes de regresar a la posición inicial.</p><p>Este ejercicio requiere una técnica adecuada para evitar lesiones en la espalda baja. Es crucial mantener el core activado y evitar movimientos bruscos durante la ejecución.</p>',
        image_path: 'back/barbell_row.png',
        original_video_path: 'back/barbell_row.mp4',
        landmarks_data_path: 'back/barbell_row_landmarks.csv'
    },
    {
        name: 'Peso Muerto',
        muscle_group: 'Espalda',
        short_description: 'Ejercicio compuesto que trabaja la espalda baja, glúteos y piernas, ideal para desarrollar fuerza y potencia en todo el cuerpo.',
        long_description: '<p>El peso muerto es uno de los ejercicios más completos en el entrenamiento de fuerza, trabajando la espalda baja, los glúteos, los isquiotibiales y los músculos estabilizadores del core. Es fundamental para desarrollar fuerza total y mejorar la postura.</p><p>Para realizarlo, el atleta se coloca frente a una barra cargada, agarra la barra con ambas manos y se levanta extendiendo caderas y rodillas mientras mantiene la barra cerca del cuerpo. Luego, regresa a la posición inicial controlando el movimiento.</p><p>Una técnica adecuada es esencial para este ejercicio. Mantén la espalda neutra, el core firme y evita redondear la zona lumbar para minimizar riesgos y maximizar beneficios.</p>',
        image_path: 'back/deadlift.png',
        original_video_path: 'back/deadlift.mp4',
        landmarks_data_path: 'back/deadlift_landmarks.csv'
    },
    {
        name: 'Remo con Mancuernas',
        muscle_group: 'Espalda',
        short_description: 'Ejercicio de tracción unilateral que trabaja los dorsales, trapecios y la espalda media, ayudando a mejorar fuerza y simetría muscular.',
        long_description: '<p>El remo con mancuernas es un ejercicio de tracción unilateral que se enfoca en el desarrollo de los dorsales, trapecios y la espalda media. También ayuda a corregir desequilibrios musculares entre ambos lados del cuerpo.</p><p>Para realizarlo, el atleta apoya una rodilla y una mano sobre un banco plano mientras sostiene una mancuerna con la mano opuesta. Luego, tira de la mancuerna hacia el torso, manteniendo el codo cerca del cuerpo y controlando el descenso.</p><p>Este ejercicio requiere mantener una postura estable y controlada durante toda la ejecución para evitar lesiones y maximizar la activación muscular.</p>',
        image_path: 'back/dumbbell_row.png',
        original_video_path: 'back/dumbbell_row.mp4',
        landmarks_data_path: 'back/dumbbell_row_landmarks.csv'
    },
    {
        name: 'Jalon al Pecho',
        muscle_group: 'Espalda',
        short_description: 'Ejercicio guiado con polea que activa los dorsales, bíceps y deltoides posteriores, ideal para principiantes y para fortalecer la espalda.',
        long_description: '<p>El jalón al pecho es un ejercicio guiado en polea que activa los dorsales, bíceps y deltoides posteriores. Es ideal tanto para principiantes como para avanzados que buscan fortalecer y definir la espalda.</p><p>Para realizarlo, el atleta se sienta frente a una máquina de polea alta y agarra la barra con ambas manos. Luego, tira de la barra hacia el pecho mientras aprieta los dorsales, antes de regresar lentamente a la posición inicial.</p><p>Es importante mantener una postura recta y evitar movimientos bruscos durante la ejecución para maximizar la activación muscular y evitar lesiones.</p>',
        image_path: 'back/lat_pull_down.png',
        original_video_path: 'back/lat_pull_down.mp4',
        landmarks_data_path: 'back/lat_pull_down_landmarks.csv'
    },
    {
        name: 'Dominadas',
        muscle_group: 'Espalda',
        short_description: 'Movimiento de peso corporal enfocado en dorsales, trapecios y bíceps; ideal para construir fuerza y resistencia en la parte superior del cuerpo.',
        long_description: '<p>Las dominadas son un ejercicio de peso corporal que trabaja principalmente los dorsales, trapecios y bíceps, además de involucrar el core como estabilizador. Son ideales para desarrollar fuerza y resistencia en la parte superior del cuerpo.</p><p>Para realizarlas, el atleta se cuelga de una barra fija con un agarre que puede ser pronado, supinado o neutro. Luego, tira del cuerpo hacia arriba hasta que la barbilla pase por encima de la barra antes de regresar lentamente a la posición inicial.</p><p>Este ejercicio requiere fuerza inicial y un control constante del movimiento. Se recomienda progresar con bandas de asistencia si eres principiante.</p>',
        image_path: 'back/pull_ups.png',
        original_video_path: 'back/pull_ups.mp4',
        landmarks_data_path: 'back/pull_ups_landmarks.csv'
    },
    {
        name: 'Curl Martillo',
        muscle_group: 'Bíceps',
        short_description: 'Variante del press militar que permite mayor rango de movimiento, trabajando cada lado del cuerpo de forma independiente para desarrollar equilibrio y simetría.',
        long_description: '<p>El curl martillo es un ejercicio enfocado en los bíceps y el braquiorradial, ofreciendo una variante que también fortalece los antebrazos. Es ideal para desarrollar fuerza y masa muscular en los brazos.</p><p>Para realizarlo, el atleta sostiene una mancuerna en cada mano con un agarre neutro. Luego, flexiona los codos para levantar las mancuernas hacia los hombros, manteniendo los codos cerca del torso. Finalmente, desciende lentamente a la posición inicial.</p><p>Es importante mantener una técnica controlada y evitar balanceos para optimizar los resultados y prevenir lesiones.</p>',
        image_path: 'biceps/hammer_curl.png',
        original_video_path: 'biceps/hammer_curl.mp4',
        landmarks_data_path: 'biceps/hammer_curl_landmarks.csv'
    },
    {
        name: 'Curl Predicador',
        muscle_group: 'Bíceps',
        short_description: 'Ejercicio fundamental para los hombros, enfocado en el deltoides anterior, con énfasis en la fuerza y estabilidad al levantar peso sobre la cabeza.',
        long_description: '<p>El curl predicador es un ejercicio que aísla los bíceps al reducir la participación de otros músculos, permitiendo un rango de movimiento controlado y efectivo. Es ideal para desarrollar la forma y la fuerza en los bíceps.</p><p>Para realizarlo, el atleta se sienta en una máquina o banco predicador, apoyando los brazos sobre una plataforma inclinada. Luego, levanta la barra o mancuerna flexionando los codos, contrayendo los bíceps en la parte superior, y regresa lentamente a la posición inicial.</p><p>Es fundamental mantener una técnica controlada y evitar utilizar el impulso para maximizar la activación muscular y prevenir lesiones.</p>',
        image_path: 'biceps/preacher_curl.png',
        original_video_path: 'biceps/preacher_curl.mp4',
        landmarks_data_path: 'biceps/preacher_curl_landmarks.csv'
    },
    {
        name: 'Cruce de Poleas',
        muscle_group: 'Pecho',
        short_description: 'Movimiento de aislamiento para el pecho, usando poleas para trabajar el pectoral mayor, ideal para definir y enfatizar la parte central del pecho.',
        long_description: '<p>El cruce de poleas es un ejercicio de aislamiento diseñado para trabajar el pectoral mayor, enfatizando la parte central del pecho. Es ideal para definir y mejorar la forma muscular del torso.</p><p>Para realizarlo, el atleta se coloca en el centro de una máquina de poleas, agarra las asas y extiende los brazos hacia adelante en un movimiento semicircular, cruzando las manos ligeramente al final. Luego, regresa lentamente a la posición inicial.</p><p>Es importante mantener los codos ligeramente flexionados y evitar balanceos para una ejecución segura y efectiva.</p>',
        image_path: 'chest/cable_crossover.png',
        original_video_path: 'chest/cable_crossover.mp4',
        landmarks_data_path: 'chest/cable_crossover_landmarks.csv'
    },
    {
        name: 'Press de Banca',
        muscle_group: 'Pecho',
        short_description: 'Movimiento clave para el desarrollo del pecho, tríceps y deltoides, realizado acostado mientras se empuja una barra hacia arriba.',
        long_description: '<p>El press de banca es un ejercicio compuesto esencial para el desarrollo del pecho, tríceps y deltoides. Es uno de los movimientos más populares para aumentar la fuerza y el tamaño muscular de la parte superior del cuerpo.</p><p>Para realizarlo, el atleta se acuesta en un banco plano y agarra una barra con ambas manos. Luego, baja la barra hacia el pecho de manera controlada y la empuja hacia arriba hasta extender completamente los brazos.</p><p>La técnica adecuada es clave para evitar lesiones. Asegúrate de mantener la espalda recta, los pies firmes en el suelo y un control constante durante el movimiento.</p>',
        image_path: 'chest/flat_bench_press.png',
        original_video_path: 'chest/flat_bench_press.mp4',
        landmarks_data_path: 'chest/flat_bench_press_landmarks.csv'
    },
    {
        name: 'Press Inclinado con Barra',
        muscle_group: 'Pecho',
        short_description: 'Ejercicio que activa principalmente la parte superior del pecho, útil para construir fuerza y tamaño en el pectoral mayor.',
        long_description: '<p>El press inclinado con barra se centra en el desarrollo de la parte superior del pecho, junto con los tríceps y los deltoides. Es ideal para aumentar la fuerza y la masa muscular en esta área específica.</p><p>Para realizarlo, el atleta se recuesta en un banco inclinado y agarra una barra con ambas manos. Luego, baja la barra hacia la parte superior del pecho y la empuja hacia arriba hasta extender los brazos completamente.</p><p>Es importante mantener una postura adecuada, con la espalda apoyada en el banco y los pies firmes en el suelo, para evitar lesiones y maximizar la efectividad.</p>',
        image_path: 'chest/incline_bench_press.png',
        original_video_path: 'chest/incline_bench_press.mp4',
        landmarks_data_path: 'chest/incline_bench_press_landmarks.csv'
    },
    {
        name: 'Extensión de Cuádriceps',
        muscle_group: 'Piernas',
        short_description: 'Movimiento en máquina que enfoca exclusivamente en los cuádriceps, ayudando a desarrollar definición y fuerza muscular en la parte frontal de las piernas.',
        long_description: '<p>La extensión de cuádriceps es un ejercicio de aislamiento que se enfoca exclusivamente en los músculos del cuádriceps, siendo ideal para mejorar la definición y la fuerza muscular en la parte frontal de las piernas.</p><p>Para realizarlo, el atleta se sienta en una máquina de extensión de piernas y coloca los tobillos debajo del rodillo acolchado. Luego, extiende las piernas hacia adelante hasta contraer completamente los cuádriceps, y regresa lentamente a la posición inicial.</p><p>Asegúrate de ajustar la máquina correctamente para un rango de movimiento adecuado y evitar estrés innecesario en las rodillas.</p>',
        image_path: 'legs/cuadriceps_extension.png',
        original_video_path: 'legs/cuadriceps_extension.mp4',
        landmarks_data_path: 'legs/cuadriceps_extension_landmarks.csv'
    },
    {
        name: 'Prensa',
        muscle_group: 'Piernas',
        short_description: 'Ejercicio de aislamiento que trabaja principalmente los cuádriceps, glúteos y isquiotibiales, promoviendo fuerza y estabilidad en las piernas.',
        long_description: '<p>La prensa es un ejercicio de aislamiento que trabaja los cuádriceps, glúteos e isquiotibiales, promoviendo fuerza y estabilidad en las piernas. Es ideal tanto para principiantes como para avanzados.</p><p>Para realizarlo, el atleta se sienta en una máquina de prensa y coloca los pies sobre la plataforma. Luego, empuja la plataforma hacia adelante hasta extender las piernas sin bloquear las rodillas, y regresa controladamente a la posición inicial.</p><p>Es crucial mantener una postura adecuada y controlar el movimiento para evitar lesiones en las articulaciones.</p>',
        image_path: 'legs/leg_press.png',
        original_video_path: 'legs/leg_press.mp4',
        landmarks_data_path: 'legs/leg_press_landmarks.csv'
    },
    {
        name: 'Curl de Isquios',
        muscle_group: 'Piernas',
        short_description: 'Ejercicio en máquina para isquiotibiales que fortalece los músculos traseros del muslo, crucial para mejorar la estabilidad y la potencia en las piernas.',
        long_description: '',
        image_path: 'legs/lying_leg_curl.png',
        original_video_path: 'legs/lying_leg_curl.mp4',
        landmarks_data_path: 'legs/lying_leg_curl_landmarks.csv'
    },
    {
        name: 'Sentadilla',
        muscle_group: 'Piernas',
        short_description: 'Ejercicio compuesto clave para el desarrollo de piernas y glúteos, activa principalmente cuádriceps, isquiotibiales y glúteos, mejorando fuerza y estabilidad corporal.',
        long_description: '<p>El curl de isquios es un ejercicio en máquina que se centra en los músculos isquiotibiales, fortaleciendo la parte trasera del muslo y mejorando la estabilidad y potencia en las piernas.</p><p>Para realizarlo, el atleta se acuesta boca abajo en la máquina de curl de piernas, coloca los tobillos debajo del rodillo acolchado y flexiona las rodillas para levantar el peso hacia los glúteos. Luego, regresa lentamente a la posición inicial.</p><p>Es importante realizar el movimiento de forma controlada y evitar hiperextender las rodillas al final del recorrido.</p>',
        image_path: 'legs/squat.png',
        original_video_path: 'legs/squat.mp4',
        landmarks_data_path: 'legs/squat_landmarks.csv'
    },
    {
        name: 'Press Militar con Barra',
        muscle_group: 'Hombros',
        short_description: 'Ejercicio fundamental para los hombros, enfocado en el deltoides anterior, con énfasis en la fuerza y estabilidad al levantar peso sobre la cabeza.',
        long_description: '<p>El press militar con barra es un ejercicio compuesto fundamental para los hombros, enfocado en el deltoides anterior, con participación de los tríceps y deltoides laterales. Es ideal para aumentar la fuerza y estabilidad del tren superior.</p><p>Para realizarlo, el atleta sostiene una barra a la altura del pecho con las manos ligeramente más separadas que los hombros. Luego, empuja la barra hacia arriba hasta extender completamente los brazos y regresa lentamente a la posición inicial.</p><p>Mantén el core activado y evita arquear la espalda para maximizar la seguridad y efectividad del movimiento.</p>',
        image_path: 'shoulders/military_press_barbell.png',
        original_video_path: 'shoulders/military_press_barbell.mp4',
        landmarks_data_path: 'shoulders/military_press_barbell_landmarks.csv'
    },
    {
        name: 'Press Militar con Mancuernas',
        muscle_group: 'Hombros',
        short_description: 'Variante del press militar que permite mayor rango de movimiento, trabajando cada lado del cuerpo de forma independiente para desarrollar equilibrio y simetría.',
        long_description: '<p>El press militar con mancuernas es una variante del press militar que permite mayor rango de movimiento y trabaja cada lado del cuerpo de forma independiente, mejorando el equilibrio y la simetría muscular.</p><p>Para realizarlo, el atleta sostiene una mancuerna en cada mano a la altura de los hombros. Luego, empuja ambas mancuernas hacia arriba hasta extender los brazos completamente y regresa controladamente a la posición inicial.</p><p>Es importante mantener una postura recta y evitar balanceos para optimizar los resultados y prevenir lesiones.</p>',
        image_path: 'shoulders/military_press_dumbbell.png',
        original_video_path: 'shoulders/military_press_dumbbell.mp4',
        landmarks_data_path: 'shoulders/military_press_dumbbell_landmarks.csv'
    },
    {
        name: 'Extensión en Polea',
        muscle_group: 'Tríceps',
        short_description: 'Variante del press militar que permite mayor rango de movimiento, trabajando cada lado del cuerpo de forma independiente para desarrollar equilibrio y simetría.',
        long_description: '<p>La extensión en polea es un ejercicio de aislamiento que se enfoca en los tríceps, ayudando a desarrollar fuerza y definición en la parte posterior de los brazos.</p><p>Para realizarlo, el atleta se coloca frente a una máquina de polea alta y agarra una barra o cuerda. Luego, extiende los brazos hacia abajo manteniendo los codos cerca del torso, y regresa lentamente a la posición inicial.</p><p>Evita mover los codos durante el ejercicio para mantener la tensión en los tríceps y maximizar la efectividad.</p>',
        image_path: 'triceps/cable_triceps_extension.png',
        original_video_path: 'triceps/cable_triceps_extension.mp4',
        landmarks_data_path: 'triceps/cable_triceps_extension_landmarks.csv'
    },
    {
        name: 'Press Francés',
        muscle_group: 'Tríceps',
        short_description: 'Ejercicio fundamental para los hombros, enfocado en el deltoides anterior, con énfasis en la fuerza y estabilidad al levantar peso sobre la cabeza.',
        long_description: '<p>El press francés es un ejercicio que aísla los tríceps, siendo ideal para aumentar la fuerza y la masa muscular en la parte posterior de los brazos.</p><p>Para realizarlo, el atleta se acuesta en un banco y sostiene una barra o mancuernas con las manos. Luego, baja el peso hacia la frente flexionando los codos y lo empuja hacia arriba hasta extender completamente los brazos.</p><p>Es esencial mantener los codos fijos y el movimiento controlado para evitar lesiones y maximizar la activación de los tríceps.</p>',
        image_path: 'triceps/french_press.png',
        original_video_path: 'triceps/french_press.mp4',
        landmarks_data_path: 'triceps/french_press_landmarks.csv'
    }
]);