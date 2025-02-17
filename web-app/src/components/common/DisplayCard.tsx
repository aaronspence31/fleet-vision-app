import React from "react";

interface DisplayCardProps {
  children?: React.ReactNode;
}

const DisplayCard: React.FC<DisplayCardProps> = ({ children }) => {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        border: "1px solid lightgray",
        borderRadius: "20px",
        padding: "20px",
        width: "100%",
        height: "100%",
        backgroundColor: "#FEFEFF",
      }}
    >
      {children}
    </div>
  );
};

export default DisplayCard;
