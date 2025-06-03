import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import GitHubLink from '../components/GitHubLink';
import InfoCard from '../components/InfoCard';


const About = () => {
    return (
        <div className='page-container'>
            <Navbar />
            <div className='main-content'>
                <div>
                    <h1 className='section-title'>Acerca de</h1>
                    <InfoCard text={
                        <>
                            <p>
                                El sistema <strong>REPS (Rating Exercise Performance System)</strong> está diseñado para ayudar a los usuarios a mejorar su técnica en ejercicios de gimnasio mediante un análisis exhaustivo y detallado de su ejecución, basado en grabaciones en video. Este sistema aprovecha tecnologías avanzadas para brindar una experiencia personalizada que guía a los usuarios hacia una ejecución más eficiente y segura de los ejercicios. La plataforma funciona como una aplicación web accesible desde cualquier navegador, donde los usuarios pueden cargar sus videos con facilidad. Una vez que el video es subido, el sistema lo procesa y genera un <em>feedback</em> técnico altamente detallado y personalizado, orientado a corregir errores, prevenir lesiones y optimizar la técnica de entrenamiento.
                            </p>

                            <p>
                                El análisis técnico de la ejecución de los ejercicios sigue una metodología bien definida que consta de las siguientes fases:
                            </p>

                            <p>
                                <strong>&emsp;1. Extracción de datos:</strong>
                                <br />&emsp;&emsp;&emsp;- En esta etapa, los datos del video son procesados utilizando el modelo de detección de pose <em>MediaPipe</em>, una herramienta avanzada capaz de identificar los puntos clave del cuerpo del usuario.
                                <br />&emsp;&emsp;&emsp;- Se generan coordenadas específicas que representan las posiciones de las articulaciones y los segmentos corporales durante el movimiento.
                                <br />&emsp;&emsp;&emsp;- A partir de esta información, se crea un "esqueleto virtual" que refleja con precisión la postura y los patrones de movimiento del usuario mientras realiza el ejercicio.
                            </p>

                            <p>
                                <strong>&emsp;2. Normalización:</strong>
                                <br />&emsp;&emsp;&emsp;- Una vez detectados los puntos clave, los esqueletos generados se redimensionan y ajustan para mantener proporciones estandarizadas, independientemente de las características físicas del usuario (altura, complexión, etc.).
                                <br />&emsp;&emsp;&emsp;- Este proceso permite una visualización uniforme y facilita la comparación entre diferentes usuarios o modelos de referencia, garantizando que las observaciones se centren en la técnica en lugar de las variaciones físicas individuales.
                            </p>

                            <p>
                                <strong>&emsp;3. Segmentación:</strong>
                                <br />&emsp;&emsp;&emsp;- Las coordenadas de los puntos clave del esqueleto se dividen en repeticiones individuales.
                                <br />&emsp;&emsp;&emsp;- Este paso implica identificar y separar cada repetición del ejercicio dentro del video cargado, permitiendo un análisis más preciso de cada ciclo de movimiento.
                                <br />&emsp;&emsp;&emsp;- La segmentación asegura que solo se utilice la información relevante para la evaluación técnica, reduciendo el ruido en los datos y simplificando el proceso de análisis.
                            </p>

                            <p>
                                <strong>&emsp;4. Sincronización:</strong>
                                <br />&emsp;&emsp;&emsp;- Las repeticiones aisladas se ajustan para igualar su duración, unificando el tiempo de ejecución de cada repetición.
                                <br />&emsp;&emsp;&emsp;- Esto garantiza una comparación coherente entre las repeticiones realizadas por el usuario y el modelo de referencia, facilitando la detección de desviaciones o inconsistencias en el ritmo, la postura y la amplitud de los movimientos.
                                <br />&emsp;&emsp;&emsp;- Este paso es fundamental para ofrecer un <em>feedback</em> claro y comprensible.
                            </p>

                            <p>
                                <strong>&emsp;5. Visualización:</strong>
                                <br />&emsp;&emsp;&emsp;- Finalmente, el sistema genera un video comparativo que combina la ejecución del usuario con una versión ideal del ejercicio realizada por un experto.
                                <br />&emsp;&emsp;&emsp;- Este video incluye superposiciones visuales, marcadores clave y comentarios gráficos que resaltan las diferencias entre ambas ejecuciones.
                                <br />&emsp;&emsp;&emsp;- Gracias a esta representación visual intuitiva, el usuario puede identificar con precisión las áreas de mejora y trabajar en la corrección de su técnica de manera efectiva.
                            </p>

                            <p>
                                En conjunto, REPS se presenta como una herramienta innovadora y completa que combina la tecnología de detección de movimiento con una interfaz práctica y amigable. El sistema no solo ayuda a los usuarios a perfeccionar su técnica y a mejorar su rendimiento físico, sino que también contribuye a la prevención de lesiones mediante la identificación de errores potencialmente peligrosos en la ejecución de los ejercicios.
                            </p>
                        </>
                    } />
                    <div className='author-section'>
                        <GitHubLink username={"Sikander035"} />
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default About;
