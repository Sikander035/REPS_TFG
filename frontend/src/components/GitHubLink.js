import React from "react";
import GitHubIcon from "@mui/icons-material/GitHub";

const GitHubLink = ({ username }) => {
    if (!username) return null;

    const handleRedirect = () => {
        window.open(`https://github.com/${username}`, "_blank", "noopener noreferrer");
    };

    return (
        <div className="github-button" onClick={handleRedirect}>
            <GitHubIcon />
            <strong>{username}</strong>
        </div>
    );
};

export default GitHubLink;
