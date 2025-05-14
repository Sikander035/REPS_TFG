import { useState } from 'react';
import { MdCloudUpload, MdDelete } from 'react-icons/md';
import { AiFillFileImage } from 'react-icons/ai';
import ReactPlayer from 'react-player';

const FileUploader = () => {
    const [image, setImage] = useState(null);
    const [fileName, setFileName] = useState("No selected file");
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [videoBlob, setVideoBlob] = useState(null);

    const handleFeedback = async (event) => {
        event.preventDefault(); // Evita la recarga de la página
        setLoading(true); // Inicia el loading

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch("http://localhost:8000/inference", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Error en la respuesta del servidor");
            }

            // Leer el stream del cuerpo de la respuesta
            const reader = response.body.getReader();
            const chunks = [];
            let done = false;

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                if (value) {
                    chunks.push(value);
                }
                done = readerDone;
            }

            // Crear un Blob a partir de los chunks
            const videoBlob = new Blob(chunks, { type: "video/mp4" });
            setVideoBlob(videoBlob); // Guardar el Blob en el estado

        } catch (error) {
            console.error("Error:", error);
            alert("Ocurrió un error al procesar el archivo");
        } finally {
            setLoading(false); // Detener el loading
        }
    };

    return (
        <>
            {fileName === "No selected file" ? (
                <form className='file-uploader'
                    onClick={() => document.querySelector(".input-field").click()}
                >
                    <input
                        type="file"
                        accept=".avi,.mp4"
                        className='input-field'
                        hidden
                        onChange={({ target: { files } }) => {
                            files[0] && setFileName(files[0].name);
                            if (files) {
                                setImage(URL.createObjectURL(files[0]));
                                setFile(files[0]);
                            }
                        }}
                    />

                    {image ?
                        <img src={image} width={150} height={150} alt={fileName} />
                        :
                        <>
                            <MdCloudUpload color='#e94e4f' size={60} />
                            <p style={{ color: 'white' }}>Browse Files to upload</p>
                        </>
                    }
                </form>
            ) : (
                <div className='uploaded-file-container'>
                    <section className='uploaded-row'>
                        <AiFillFileImage color='#e94e4f' />
                        <span className='upload-content'>
                            <span style={{ color: 'white', fontSize: '1rem' }}>{fileName}</span>
                            <MdDelete color='#e94e4f'
                                style={{ marginLeft: '10px' }}
                                onClick={() => {
                                    setFileName("No selected file");
                                    setImage(null);
                                    setFile(null);
                                    setVideoBlob(null); // Limpia el estado del video
                                }}
                            />
                        </span>
                    </section>

                    {!videoBlob ? (
                        <>
                            {loading ? (
                                <>
                                    <div className="loading-spinner"></div>
                                    <p style={{ color: 'white' }}>Procesando archivo...</p>
                                </>
                            ) : (
                                <button className='uploaded-file-button' onClick={handleFeedback}>
                                    Obtener feedback
                                </button>
                            )}
                        </>
                    ) : (
                        <div className='feedback-video-container'>
                            <ReactPlayer
                                className='feedback-video-player'
                                url={URL.createObjectURL(videoBlob)} // Crea una URL temporal desde el Blob
                                controls
                                width="100%"
                            />
                        </div>
                    )}
                </div>
            )}
        </>
    );
};

export default FileUploader;
