import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MdCloudUpload, MdDelete } from 'react-icons/md';
import { AiFillFileImage } from 'react-icons/ai';

const FileUploader = () => {
    const [image, setImage] = useState(null);
    const [fileName, setFileName] = useState("No selected file");
    const [file, setFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const navigate = useNavigate();

    const handleFeedback = async (event) => {
        event.preventDefault();
        
        // Verificar que hay un archivo seleccionado
        if (!file) {
            alert("Por favor selecciona un archivo primero");
            return;
        }

        setIsUploading(true);
        setUploadError(null);

        try {
            console.log('🚀 Starting file upload and analysis...');
            
            // Determinar ejercicio basado en URL actual
            const urlParams = new URLSearchParams(window.location.search);
            const exerciseParam = urlParams.get('exercise');
            const exerciseName = exerciseParam || 'military_press_dumbbell';
            
            console.log('📝 Exercise name:', exerciseName);
            console.log('📁 File:', file.name, `(${(file.size / 1024 / 1024).toFixed(2)} MB)`);

            // Crear FormData para enviar el archivo
            const formData = new FormData();
            formData.append('file', file);

            // Hacer petición a la API
            console.log('📤 Uploading to API...');
            const response = await fetch(
                `http://localhost:8000/analyze-exercise?exercise_name=${exerciseName}`, 
                {
                    method: 'POST',
                    body: formData,
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Error ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('✅ Upload successful:', result);
            
            // Navegar a la página de feedback con el job_id
            const navigationState = {
                jobId: result.job_id,
                exerciseName: formatName(exerciseName),
                fileName: fileName,
                uploadedAt: new Date().toISOString()
            };
            
            console.log('🧭 Navigating to feedback with state:', navigationState);
            
            navigate('/feedback', {
                state: navigationState
            });

        } catch (error) {
            console.error('❌ Error uploading file:', error);
            setUploadError(error.message);
        } finally {
            setIsUploading(false);
        }
    };

    const resetUploader = () => {
        setFileName("No selected file");
        setImage(null);
        setFile(null);
        setUploadError(null);
    };

    const formatName = (name) => {
        return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    };

    const handleFileSelect = ({ target: { files } }) => {
        if (files[0]) {
            console.log('📁 File selected:', files[0].name);
            setFileName(files[0].name);
            setImage(URL.createObjectURL(files[0]));
            setFile(files[0]);
            setUploadError(null);
        }
    };

    return (
        <>
            {fileName === "No selected file" ? (
                <form 
                    className='file-uploader'
                    onClick={() => !isUploading && document.querySelector(".input-field").click()}
                    style={{ opacity: isUploading ? 0.6 : 1 }}
                >
                    <input
                        type="file"
                        accept=".avi,.mp4,.mov"
                        className='input-field'
                        hidden
                        disabled={isUploading}
                        onChange={handleFileSelect}
                    />

                    {isUploading ? (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '15px' }}>
                            <div className="loading-spinner"></div>
                            <p style={{ color: 'white', margin: 0 }}>Subiendo archivo...</p>
                            <small style={{ color: 'var(--font-color)', opacity: 0.7 }}>
                                Este proceso puede tomar unos minutos
                            </small>
                        </div>
                    ) : image ? (
                        <img src={image} width={150} height={150} alt={fileName} />
                    ) : (
                        <>
                            <MdCloudUpload color='#e94e4f' size={60} />
                            <p style={{ color: 'white' }}>Browse Files to upload</p>
                            <small style={{ color: 'var(--font-color)', opacity: 0.7 }}>
                                Formatos: MP4, AVI, MOV
                            </small>
                        </>
                    )}
                </form>
            ) : (
                <div className='uploaded-file-container'>
                    <section className='uploaded-row'>
                        <AiFillFileImage color='#e94e4f' />
                        <span className='upload-content'>
                            <span style={{ color: 'white', fontSize: '1rem' }}>{fileName}</span>
                            {!isUploading && (
                                <MdDelete 
                                    color='#e94e4f'
                                    style={{ marginLeft: '10px', cursor: 'pointer' }}
                                    onClick={resetUploader}
                                    title="Eliminar archivo"
                                />
                            )}
                        </span>
                    </section>

                    {uploadError && (
                        <div style={{ 
                            color: '#ff6b6b', 
                            fontSize: '14px', 
                            marginBottom: '15px',
                            textAlign: 'center',
                            padding: '12px',
                            background: 'rgba(255, 107, 107, 0.1)',
                            borderRadius: '6px',
                            border: '1px solid rgba(255, 107, 107, 0.3)'
                        }}>
                            <strong>❌ Error:</strong> {uploadError}
                            <br />
                            <small>Verifica que el servidor esté corriendo y el archivo sea válido</small>
                        </div>
                    )}

                    <button 
                        className='uploaded-file-button' 
                        onClick={handleFeedback}
                        disabled={isUploading}
                        style={{ 
                            opacity: isUploading ? 0.6 : 1,
                            cursor: isUploading ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '8px'
                        }}
                    >
                        {isUploading ? (
                            <>
                                <div className="loading-spinner" style={{ width: '16px', height: '16px' }}></div>
                                Iniciando análisis...
                            </>
                        ) : (
                            '🚀 Obtener feedback'
                        )}
                    </button>
                    
                    {!isUploading && (
                        <small style={{ 
                            color: 'var(--font-color)', 
                            opacity: 0.7, 
                            textAlign: 'center',
                            marginTop: '10px',
                            display: 'block'
                        }}>
                            El análisis puede tomar varios minutos
                        </small>
                    )}
                </div>
            )}
        </>
    );
};

export default FileUploader;