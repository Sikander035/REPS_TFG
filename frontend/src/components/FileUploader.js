import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MdCloudUpload, MdDelete } from 'react-icons/md';
import { AiFillFileImage } from 'react-icons/ai';

const FileUploader = () => {
    const [image, setImage] = useState(null);
    const [fileName, setFileName] = useState("No selected file");
    const [file, setFile] = useState(null);
    const navigate = useNavigate();

    const handleFeedback = (event) => {
        event.preventDefault();
        
        // Verificar que hay un archivo seleccionado
        if (!file) {
            alert("Por favor selecciona un archivo primero");
            return;
        }
        
        // Solo navegar a la página de feedback con el archivo
        navigate('/feedback', {
            state: {
                fileName: fileName,
                originalFile: file
            }
        });
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
                                }}
                            />
                        </span>
                    </section>

                    <button className='uploaded-file-button' onClick={handleFeedback}>
                        Obtener feedback
                    </button>
                </div>
            )}
        </>
    );
};

export default FileUploader;