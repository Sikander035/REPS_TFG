import React from "react";
import { useNavigate } from "react-router-dom";

const ForgotPasswordCard = () => {
    const navigate = useNavigate();

    const handleForgotPassword = (event) => {
        event.preventDefault(); // Evita la recarga de la página
        const email = event.target[0].value;
        const repeatEmail = event.target[1].value;

        if (!checkEmail(email, repeatEmail)) {
            deleteTextFields(event);
            return;
        }

        fetch("http://localhost:8000/forgot-password", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ email }),
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


    function checkEmail(email, repeatEmail) {
        if (email !== repeatEmail) {
            alert("Los email no coinciden");
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
            <div className="auth-card-form-container auth-card-forgot-password">
                <form onSubmit={handleForgotPassword}>
                    <h2>Restablece tu contraseña</h2>
                    <input type="email" placeholder="Email" />
                    <input type="email" placeholder="Repeat email" />
                    <button type="submit">ENVIAR CORREO</button>
                </form>
            </div>
            <div className="auth-card-toggle-container-forgot-password">
                <div className="auth-card-toggle">
                    <div className="auth-card-toggle-panel auth-card-toggle-right">
                        <h1>¡Atención!</h1>
                        <p>Se le enviará un correo electrónico con las instrucciones para cambiar su contraseña.</p>
                    </div>
                </div>
            </div>
        </div >
    );
};

export default ForgotPasswordCard;
