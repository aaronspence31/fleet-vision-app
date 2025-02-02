/* eslint-disable @next/next/no-img-element */
// Wrapper component for the signin page

"use client";
import React, { useState, useEffect, useRef, useReducer } from "react";
import { Box, Typography, Divider } from "@mui/material";
import styles from "./page.module.css";
import { LiveFrame, Event } from "./types";
import Image from "next/image";

export default function BodyDemo() {
  const [livePredictStream, setLivePredictStream] = useState<LiveFrame[]>([]);

  const [events, setEvents] = useState<Event[]>([]);

  const bodyImgRef = useRef(null);

  useEffect(() => {
    const eventSource = new EventSource(`http://127.0.0.1:5000/body_stream`);
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (bodyImgRef.current) {
        bodyImgRef.current.src = `data:image/jpg;base64,${data.image}`;
      }
      addLivePredictions(data.predictions, data.first_frame_num);
      if (data.event != 0) {
        addEvent(data.event.replaceAll("'", '"'));
      }
    };
    return () => {
      eventSource.close();
    };
  }, []);

  // useEffect(() => {
  //   if(events.length > 1){
  //     const prevEvent = {...(events[1])}
  //     const currEvent = {...(events[0])}
  //     if (prevEvent.label == currEvent.label && currEvent.frameStart - prevEvent.frameEnd < 15){
  //       prevEvent.frameEnd = currEvent.frame
  //     }
  //   }
  // }, [events])

  const addEvent = (action_event: any) => {
    console.log(livePredictStream.length);
    const transformedEvent = {
      ...JSON.parse(action_event),
      camera: "Face",
    };

    // if (parseInt(transformedEvent.cont)) {
    //   console.log("PrevEvent: ", prevEvent);
    //   let newEvent = { ...prevEvent };
    //   newEvent.frameEnd = transformedEvent.frameEnd;
    //   setEvents((events) => [newEvent].concat(events.slice(1)));
    // } else {
    delete transformedEvent.cont;
    setEvents((events) => [transformedEvent].concat(events));
    // }
  };

  const addLivePredictions = (predictions: string[], startingFrame: string) => {
    const transformPredictions: LiveFrame[] = predictions.map(
      (prediction, index) => ({
        value: prediction,
        frameNum: parseInt(startingFrame) + index,
        camera: "Face",
      })
    );
    transformPredictions.reverse();

    setLivePredictStream((livePredictStream) =>
      [...transformPredictions].concat(livePredictStream)
    );
  };

  const removeLivePredictions = () => {
    setLivePredictStream((livePredictStream) => livePredictStream.slice(0, 30));
  };

  useEffect(() => {
    if (livePredictStream.length > 150) {
      removeLivePredictions();
    }
  }, [livePredictStream]);

  return (
    <Box className={styles.pageContainer}>
      <Box className={styles.cameraContainer}>
        <Box>
          <Typography variant="h6" textAlign="center">
            Body Cam
          </Typography>
          <Box className={styles.cameraWrapper}>
            <Image
              ref={bodyImgRef}
              src={""}
              className={styles.video}
              width={300}
              height={300}
              alt="Live Stream"
            />
          </Box>
        </Box>
      </Box>
      <Box className={styles.listContainer}>
        <Box className={styles.liveClasContainer}>
          <Typography variant="h5" textAlign={"center"} fontWeight={500}>
            Event Feed
          </Typography>
          {events.map((event, index) => (
            <Box key={index} className={styles.liveClasBox}>
              <Typography
                textAlign={"center"}
                width={150}
                textTransform={"capitalize"}
              >
                {event.label.replaceAll("_", " ")}
              </Typography>
              <Divider orientation="vertical" flexItem />
              <Typography
                textAlign={"center"}
                width={150}
              >{`From: Frame #${event.frameStart}`}</Typography>
              <Divider orientation="vertical" flexItem />
              <Typography
                textAlign={"center"}
                width={150}
              >{`To: Frame #${event.frameEnd}`}</Typography>
            </Box>
          ))}
        </Box>
        <Box className={styles.liveClasContainer}>
          <Typography variant="h5" textAlign={"center"} fontWeight={500}>
            Live Frame Classification
          </Typography>
          {livePredictStream.map((prediction, index) => (
            <Box key={index} className={styles.liveClasBox}>
              <Typography
                textAlign={"center"}
                width={150}
              >{`Frame #${prediction.frameNum}`}</Typography>
              <Divider orientation="vertical" flexItem />
              <Typography textAlign={"center"} width={150}>
                {prediction.value}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
