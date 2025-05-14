Este directorio contiene todos los elementos necesarios para el despliegue de la `base de datos` del proyecto.

La `base de datos` es un componente fundamental del sistema que almacena y gestiona toda la información necesaria para el funcionamiento de la aplicación de manera eficiente y segura. Utiliza MongoDB, una base de datos *NoSQL* de documentos que ofrece una gran escalabilidad y flexibilidad, y un modelo de consultas e indexación avanzado.

# 📂 Estructura

Para el desarrollo de la `base de datos` se ha seguido la siguiente estructura:

```
database/
├── .env                    # Variables de entorno
├── docker-compose.yml      # Configuración de contenedor Docker
├── init-mongo.js           # Script de inicialización de la base de datos MongoDB
└── README.md               # Documentación completa del proyecto

```

## 🛠️ Variables de entorno

El archivo `.env` se utiliza para definir variables de entorno que son esenciales para la configuración del entorno de desarrollo. Contiene información sensible como las credenciales de acceso a la base de datos.

A continuación, se muestra como debe ser la estructura del archivo `.env` para poder desplegar correctamente el proyecto:

```
MONGO_INITDB_ROOT_USERNAME=username
MONGO_INITDB_ROOT_PASSWORD=password
```


> [!NOTE]
> Esto es solo un ejemplo, modifica las credenciales como desees.


## 🚀 Script de Inicialización

El archivo `init-mongo.js` es el encargado de realizar la configuración inicial de la base de datos **MongoDB**. Este script se ejecuta al iniciar el contenedor **Docker** y establece las colecciones, índices y documentos necesarios para comenzar a utilizar la base de datos. Este proceso garantiza que, al desplegar la base de datos, se cargue información importante automáticamente. 

# 📦 Collections

En **MongoDB**, las collections son estructuras que agrupan documentos similares. Son equivalentes a las tablas en bases de datos relacionales pero no imponen un esquema o estructura rígida para guardar información.

En el archivo `init-mongo.js` se definen las colecciones que se van a utilizar en la base de datos para almacenar los datos sobre los usuarios y sobre los ejercicios.

A continuación, se describen más en detalle cada una de las colecciones empleadas.

### 👤 Users

La coleccion `users` almacena los datos de los usuarios registrados en el sistema. Permite gestionar la autenticación, el cambio de credenciales y el perfil de los usuarios. Cada documento de esta colección representa un usuario y contiene la siguiente información:

- `name`: Es un campo requerido y sirve para almacenar el nombre del usuario.
- `email`: Es un campo requerido y sirve para almacenar el correo electrónico del usuario. Además, se garantiza que no haya dos usuarios con el mismo correo electrónico.
- `password`: Es un campo requerido y sirve para almacenar la contraseña del usuario.

### 🏋️‍♂️Exercises

La colección `exercises` contiene los datos de los ejercicios disponibles en el sistema. Esta colección permite almacenar la información necesaria para que el usuario pueda entender la técnica del ejercicio gracias a su descripción y a los recursos multimedia asociados a este. Cada documento de esta colección representa un ejercicio y contiene la siguiente información:

- **`name`**: Es un campo requerido y sirve para guardar el nombre del ejercicio. Además, se garantiza que no haya dos ejercicios con el mismo nombre.
- **`muscle_group`**: Es un campo requerido y sirve para almacenar el grupo muscular que se trabaja en el ejercicio.
- **`short_description`**: Es un campo requerido y sirve para guardar una descripción breve del ejercicio.
- **`long_description`**: Es un campo requerido y sirve para almacenar una descripción detallada del ejercicio.
- **`image_path`**: Es un campo requerido y sirve para guardar la ruta de la imagen asociada al ejercicio.
- **`original_video_path`**: Es un campo requerido y sirve para guardar la ruta del video original que muestra cómo realizar el ejercicio.
- **`landmarks_data_path`**: Es un campo requerido y sirve para almacenar la ruta del archivo que contiene los puntos de referencia obtenidos con **mediapipe** para guiar la realización del ejercicio.
