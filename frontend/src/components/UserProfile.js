import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const UserProfile = () => {
    const [isOpen, setIsOpen] = useState(false); // Estado para abrir/cerrar el menú
    const navigate = useNavigate();

    // Abre o cierra el menú al hacer clic en el avatar
    const toggleMenu = () => {
        setIsOpen(!isOpen);
    };

    // Lógica para cerrar sesión (puedes añadir funcionalidad real aquí)
    const handleLogout = () => {
        navigate('/'); // Redirigir a la página de inicio
        setIsOpen(false); // Cerrar el menú al hacer clic en cerrar sesión
    };

    return (
        <div className="user-profile">
            <div className="avatar" onClick={toggleMenu}>
                <img
                    src={require('../assets/default_profile.svg').default} // Reemplaza con la imagen del perfil
                    alt="Perfil"
                />
            </div>

            {/* Menú desplegable */}
            {isOpen && (
                <div className="dropdown-menu">
                    <button className="dropdown-item" onClick={() => console.log('Ver perfil')}>
                        Ver Perfil
                    </button>
                    <button className="dropdown-item" onClick={handleLogout}>
                        Cerrar sesión
                    </button>
                </div>
            )}
        </div>
    );
};

export default UserProfile;
