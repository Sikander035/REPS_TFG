## Estilos de Commit

Este proyecto sigue el estándar de Conventional Commits para mantener un historial de commits claro y organizado. A continuación, se describen los tipos de commit principales que puedes usar:

- **feat**: Para añadir una nueva funcionalidad al proyecto.
  - **Ejemplo**: `feat: add user authentication module`

- **fix**: Para corregir errores o bugs en el código.
  - **Ejemplo**: `fix: correct password validation logic`

- **chore**: Para tareas de mantenimiento o configuraciones que no afectan la funcionalidad del proyecto.
  - **Ejemplo**: `chore: update dependencies`

- **docs**: Para cambios en la documentación, como archivos README, comentarios, o documentos externos.
  - **Ejemplo**: `docs: update API usage section in README`

- **style**: Para ajustes de estilo que no afectan la funcionalidad, como formateo o espacios.
  - **Ejemplo**: `style: fix indentation in config file`

- **refactor**: Para refactorizar el código sin cambiar su comportamiento externo.
  - **Ejemplo**: `refactor: optimize image processing function`

- **perf**: Para mejoras de rendimiento en el código.
  - **Ejemplo**: `perf: reduce memory usage in data processing`

- **test**: Para añadir o mejorar pruebas unitarias, de integración u otras.
  - **Ejemplo**: `test: add unit tests for user registration`

- **build**: Para cambios en el sistema de construcción o configuración del proyecto.
  - **Ejemplo**: `build: update Docker configuration`

- **ci**: Para configuraciones o scripts de integración continua (CI) y despliegue.
  - **Ejemplo**: `ci: configure GitHub Actions for automated testing`

- **revert**: Para deshacer o revertir un commit previo.
  - **Ejemplo**: `revert: revert previous commit on user permissions fix`
 
## Manual de despliegue
Inicialmente, tendremos que definir un archivo .env en la raíz del proyecto. Este archivo contendrá información como el usuario, contraseña y nombre de la base de datos, el correo que enviará los emails de restablecimiento de contraseña y una clave secreta generada en el panel de control de Google para poder enviar correos electrónicos. 
```
MONGO_INITDB_ROOT_USERNAME=username
MONGO_INITDB_ROOT_PASSWORD=password
EMAIL_PASSWORD=service-password
EMAIL_SENDER=email@email.com
MONGO_DB_NAME=database-name
```
Posteriormente, levantaremos los contenedores
```
docker-compose up
```

