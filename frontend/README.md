Este directorio contiene todos los elementos necesarios para el despliegue del `frontend` del proyecto. 

El `frontend` se refiere a la parte visible y ubicada en el lado del cliente de un sitio web o aplicación con la que los usuarios pueden interactuar directamente. Es responsable de presentar los datos de manera visual y atractiva, permitiendo a los usuarios interactuar con el sistema de forma efectiva.

# 📂 Estructura

Para la creación del `frontend`, se hizo uso de `React`, una biblioteca de código abierto de `JavaScript` diseñada para crear interfaces de usuario con el objetivo de facilitar el desarrollo de aplicaciones en una sola página. 

Además, se ha hecho uso de `Create React App`, una herramienta que configura automáticamente un entorno de desarrollo, optimizando el proceso de creación del proyecto. Esta herramienta genera la estructura inicial del proyecto, instala las dependencias necesarias y facilita el despliegue entre muchas otras cosas.

La estructura seguida en este proyecto es la siguiente:

```
frontend/
├── node_modules                # Dependencias del proyecto
├── public/                     # Archivos públicos (favicon, index.html, etc.)
├── src/                        # Código fuente del proyecto
│   ├── assets/                 # Recursos estáticos como imágenes
│   ├── components/             # Componentes reutilizables
│   ├── pages/                  # Páginas principales de la aplicación
│   ├── styles/                 # Archivos de estilos CSS
│   ├── App.js                  # Punto de entrada de la aplicación
│   ├── index.js                # Archivo principal para ReactDOM
│   ├── reportWebVitals.js      # Métricas de rendimiento
├── package.json                # Dependencias y scripts del proyecto
├── package-lock.json           # Archivo de bloqueo para npm
├── Dockerfile                  # Configuración para construir la imagen Docker
└── README.md                   # Documentación completa
```

# 🧩 Componentes

Una de las principales características de `React` es su enfoque basado en componentes. React permite construir interfaces de usuario a partir de componentes reutilizables, lo que facilita la organización, el mantenimiento y la modularidad de las aplicaciones.

Los componentes en React son bloques de construcción independientes y reutilizables que encapsulan una parte de la interfaz de usuario. Cada componente tiene su propia lógica y apariencia, lo que permite crear interfaces complejas a partir de piezas más pequeñas y manejables.

## 🔐 AuthCard

El componente `AuthCard` es responsable de manejar la autenticación de usuarios. Está compuesto por dos formularios:

- **Formulario de Login**: Permite a los usuarios iniciar sesión con una cuenta previamente creada. Cuando el usuario introduce sus credenciales (correo electrónico y contraseña), se realiza una petición **POST** al endpoint `/login`de la API. Si la autenticación es exitosa, el usuario será redirigido a la pantalla de `Home`.
- **Formulario de Registro**: Permite a los usuarios registrarse en `REPS` mediante un correo electrónico que no haya sido registrado, un nombre y una contraseña. Cuando el usuario introduce los datos de su cuenta, se realiza una petición **POST** al endpoint `/register`de la API. Una vez registrada la cuenta, el usuario podrá iniciar sesión a través del **formulario de login**.

## 🏋️ ExerciseCard

El componente `ExerciseCard` se utiliza para mostrar información detallada sobre ejercicios físicos. Cada tarjeta incluye los siguientes datos:

- Nombre del ejercicio
- Descripción corta del ejercicio
- Imagen del ejercicio

Además, se ha añadido el botón *Probar ahora*. Cuando el usuario pulse en él, será redirigido a una nueva página en la que podrá subir su video para obtener **feedback** sobre su técnica.

La información mostrada en este componente se obtiene mediante una petición **GET** al endpoint `/exercises?exercise=${nombre_del_ejercicio}` de la API.

En cuanto a la imagen del ejercicio, se obtiene igualmente mediante una petición **GET** al endpoint `/image?image_name=${imagePath}` de la API.

## 📹 ExerciseVideoCard

El componente `ExerciseVideoCard` sirve para mostrar información un poco más detallada sobre el ejercicio que la presentada en el componente `ExerciseCard`. Además, se incluye un vídeo sobre la ejecución técnica del ejercicio para ayudar al usuario a identificarlo visualmente.

La información mostrada en este componente se obtiene mediante una petición **GET** al endpoint `/exercises?exercise=${nombre_del_ejercicio}` de la API. 

Además, el video también se obtiene mediante una petición **GET** al endpoint `/video?video_name=${video_path}` de la API.

## 📤 FileUploader

El componente `FileUploader` gestiona la subida de archivos de video al sistema. Proporciona:

- Interfaz para subir videos desde el almacenamiento local
- Validación de formato de archivo (.avi o .mp4)
- Interfaz para eliminar archivos eliminados

## 👣 Footer

El componente `Footer` aparece en la parte inferior de todas las páginas. Contiene información sobre los creadores de `REPS`.

## 🔑 ForgotPasswordCard

El componente `ForgotPasswordCard` permite a los usuarios recuperar su contraseña. Presenta una interfaz similar al componente `AuthCard` . 

Para ello, este componente solicita al usuario un correo electrónico de recuperación para poder restablecer la contraseña de su cuenta de forma segura. 

> [!NOTE]
> El procedimiento de recuperación de contraseñas está explicado en la documenta> ción del backend

## 💻 GithubLinks

El componente `GithubLinks` sirve para proporcionar al usuario un enlace al perfil de los creadores de `REPS`.

## ℹ️ InfoCard

El componente `InfoCard` se utiliza para mostrar cualquier tipo de información al usuario. Principalmente, este componente se utiliza en la página de `About`, en la que se incluye información sobre el proyecto.

## 🔝 Navbar

El componente `Navbar` es la barra de navegación principal que contiene los siguientes elementos:

- Nombre del proyecto
- Enlaces a las páginas principales
- Menú para ver el perfil o cerrar sesión (componente `UserProfile`)

## 🔄 ResetPasswordCard

El componente `ResetPasswordCard` permite a los usuarios restablecer su contraseña. Presenta una interfaz similar al componente `AuthCard` . 

Este componente se utiliza en la página `/reset-password?token={token]` , a la que se puede acceder mediante el enlace recibido por correo electrónico. 

Se le solicitará al usuario una nueva contraseña y será enviada mediante una petición **POST** al endpoint `/reset-password` de la API incluyendo el **token** y la **nueva contraseña**. En el caso de que se cumplan una serie de requisitos, la contraseña del usuario será modificada.

> [!NOTE]
> El procedimiento de recuperación de contraseñas está explicado en la documenta> ción del backend

## ⬆️ UploadCard

El componente `UploadCard` proporciona una interfaz para la subida de videos. Se utiliza como contenedor del componente `FileUploader` y presenta una interfaz intuitiva para el usuario.

## 👤 UserProfile

El componente `UserProfile` consiste en un pequeño menú con dos opciones:

- Ver perfil del usuario
- Cerrar sesión

# 📄 Páginas

## 🏠 Home

La página `Home` es la página principal de la aplicación. Se muestra después de que el usuario inicie sesión correctamente. Está página contiene varias `ExerciseCard` organizadas según el grupo muscular principal que se trabaje.

Para obtener está información, se realiza una petición **GET** al endpoint `/exercises` de la API. Una vez que se dispone de esta información, se manipula la información, se filtra por grupo muscular y se muestra.

## 🏋️ Exercises

La página `Exercises` muestra información más específica sobre el ejercicio, concretamente la siguiente:

- Nombre del ejercicio
- Descripción detallada
- Video de referencia para el usuario

Además, se incluye el componente `UploadCard` para permitir al usuario subir sus videos para evaluar su técnica.

## ℹ️ About

La página `About` proporciona información general sobre el proyecto `REPS`.

## ❓ FAQ

La página `FAQ` (Frequently Asked Questions) contiene respuestas a las preguntas más comunes sobre el uso de la plataforma. 

> [!IMPORTANT]
> Actualmente no está desarrollada

## 🔑 Login

La página `Login` es el punto de entrada para los usuarios registrados. Utiliza el componente `AuthCard` para:

- Permitir el inicio de sesión de usuarios existentes
- Ofrecer la opción de registro para nuevos usuarios
- Proporcionar un enlace a la página `ForgotPassword`

## 🔄 ForgotPassword

La página `ForgotPassword` implementa el componente `ForgotPasswordCard` para el proceso de recuperación de contraseña. Su objetivo principal es proporcionar al usuario una interfaz para introducir un correo electrónico válido para la recuperación de sus credenciales.

## 🔒 ResetPassword

La página `ResetPassword` utiliza el componente `ResetPasswordCard` para completar el proceso de restablecimiento de contraseña. Esta paǵina es accesible a través del enlace recibido por correo electrónico.
