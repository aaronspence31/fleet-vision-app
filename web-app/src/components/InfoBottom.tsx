"use client";
import { useEffect, useRef } from "react";
import * as d3 from "d3";

const InfoBottom = ({ tripDistance }: { tripDistance: number }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    //@ts-ignore
    containerRef.current.innerHTML = "";
    generate(containerRef.current);
  }, [tripDistance]);

  const getCurrentTime = () => {
    const now = new Date();
    let hours = now.getHours() % 12 || 12; // Convert 0 to 12 for AM/PM format
    let minutes = now.getMinutes().toString().padStart(2, "0");
    let ampm = now.getHours() >= 12 ? "PM" : "AM";

    return `${hours}:${minutes} ${ampm}`;
  };

  //@ts-ignore
  const generate = (el) => {
    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", "100%")
      .attr("height", "100%");

    const g = svg.append("g");

    // Trip km text
    g.append("text")
      .text(`Trip: ${tripDistance}km`)
      .attr("x", "90px")
      .attr("y", "40px")
      .attr("font-size", "18")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // Red line
    g.append("image")
      .attr("href", "/images/red-line.svg")
      .attr("x", "160px")
      .attr("y", "22px")
      .attr("width", "25px")
      .attr("height", "25px");

    // Hour
    g.append("text")
      .text(getCurrentTime())
      .attr("x", "230px")
      .attr("y", "40px")
      .attr("font-size", "18")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // Red line
    g.append("image")
      .attr("href", "/images/red-line.svg")
      .attr("x", "280px")
      .attr("y", "22px")
      .attr("width", "25px")
      .attr("height", "25px");

    // Mileage
    g.append("text")
      .text("Waterloo, ON")
      .attr("x", "370px")
      .attr("y", "40px")
      .attr("font-size", "18")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // White line
    g.append("image")
      .attr("href", "/images/white-line.svg")
      .attr("x", "65px")
      .attr("y", "70px")
      .attr("width", "320px")
      .attr("height", "10px");

    // iPhone text
    g.append("text")
      .text("iPhone")
      .attr("x", "120px")
      .attr("y", "100px")
      .attr("font-size", "16")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // Musical note
    g.append("image")
      .attr("href", "/images/musical-note.svg")
      .attr("x", "170px")
      .attr("y", "80px")
      .attr("width", "25px")
      .attr("height", "25px");

    // Song text
    g.append("text")
      .text("ABBA - Waterloo")
      .attr("x", "285px")
      .attr("y", "97px")
      .attr("font-size", "12")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");
  };

  return <div className="info-bottom" ref={containerRef}></div>;
};

export default InfoBottom;
