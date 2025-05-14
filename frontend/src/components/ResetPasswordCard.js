import React from "react";
import { useNavigate } from "react-router-dom";

const ResetPasswordCard = () => {
    const navigate = useNavigate();

    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get("token");

    const handleResetPassword = (event) => {
        event.preventDefault(); // Evita la recarga de la página
        const password = event.target[0].value;
        const new_password = event.target[1].value;

        if (!checkCredentials(password, new_password)) {
            deleteTextFields(event);
            return;
        }

        fetch("http://localhost:8000/reset-password", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ token, new_password }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Success:", data);
                if (data.success) {
                    navigate("/");
                } else {
                    alert("Error: " + data.detail);
                }
            })
            .catch((error) => {
                console.error("Error:", error);
            });

        // Elimina el contenido de los campos de texto
        deleteTextFields(event);
    };

    function checkCredentials(password, repeatPassword) {
        if (password !== repeatPassword) {
            alert("Las contraseñas no coinciden");
            return false;
        }
        return true;
    }

    function deleteTextFields(event) {
        event.target[0].value = "";
        event.target[1].value = "";
    }

    return (
        <div className="auth-card-container">
            <div className="auth-card-form-container auth-card-reset-password">
                <form onSubmit={handleResetPassword}>
                    <h2>Restablece tu contraseña</h2>
                    <input type="password" placeholder="Password" />
                    <input type="password" placeholder="Repeat password" />
                    <button type="submit">CAMBIAR CONTRASEÑA</button>
                </form>
            </div>
            <div className="auth-card-toggle-container-reset-password">
                <div className="auth-card-toggle">
                    <div className="auth-card-toggle-panel auth-card-toggle-right">
                        <h1>¡Atención!</h1>
                        <p>Este proceso modificará la contraseña de tu cuenta. Por favor, asegúrate de recordar la nueva contraseña.</p>
                    </div>
                </div>
            </div>
        </div >
    );
};

export default ResetPasswordCard;
