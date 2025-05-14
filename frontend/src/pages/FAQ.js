import React from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import FAQCard from '../components/FAQCard';

const FAQ = () => {

    return (
        <div className='page-container'>
            <Navbar />
            <div className='main-content'>
                <div>
                    <h1 className='section-title'>Preguntas frecuentes</h1>
                </div>
                <div className='FAQ-question-container'>
                    <div>
                        <FAQCard question={"¿Qué es REPS y para qué sirve?"} answer={"REPS (Rating Exercise Performance System) es una herramienta diseñada para ayudar a los usuarios a mejorar su técnica de ejercicios de gimnasio. El sistema analiza los movimientos capturados en un video y proporciona feedback detallado para corregir errores y prevenir lesiones."}></FAQCard>
                        <FAQCard question={"¿Cómo funciona el sistema?"} answer={"El usuario sube un video grabado mientras realiza un ejercicio. El sistema utiliza tecnología de visión por computador para analizar la postura y los movimientos, ofreciendo un análisis con sugerencias para mejorar la técnica."}></FAQCard>
                        <FAQCard question={"¿Qué tipo de dispositivo necesito para grabar los videos?"} answer={"Puedes grabar los videos con cualquier smartphone, tablet o cámara que tenga una calidad de grabación estándar. Asegúrate de que el video capture todo el cuerpo y el movimiento completo del ejercicio."}></FAQCard>
                        <FAQCard question={"¿REPS es compatible con mi entrenador personal?"} answer={"Por supuesto. Los informes generados por el sistema pueden ser compartidos con tu entrenador para que juntos trabajen en la mejora de tu técnica."}></FAQCard>
                    </div>
                    <div>
                        <FAQCard question={"¿Qué tipo de ejercicios puedo analizar con REPS?"} answer={"Actualmente, el sistema esta diseñado para un único ejercicio de gimnasio, el press militar con mancuernas. Se están desarrollando nuevas funcionalidades para incluir más tipos de ejercicios."}></FAQCard>
                        <FAQCard question={"¿Es necesario tener experiencia en el gimnasio para usar REPS?"} answer={"No, REPS está diseñado tanto para principiantes que quieren aprender la técnica adecuada como para usuarios avanzados que buscan perfeccionar sus movimientos."}></FAQCard>
                        <FAQCard question={"¿Cómo puedo acceder a los resultados y al feedback?"} answer={"Tras analizar el video, el sistema generará un informe visual que podrás ver directamente en tu dispositivo. También tendrás la opción de descargar los resultados para guardarlos o compartirlos con un entrenador."}></FAQCard>
                        <FAQCard question={"¿Qué debo hacer si no obtengo resultados claros en mi análisis?"} answer={"Asegúrate de que el video esté bien iluminado y que tu postura sea completamente visible durante el ejercicio. Si el problema persiste, puedes contactar con nuestro equipo técnico para resolverlo."}></FAQCard>
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default FAQ;
