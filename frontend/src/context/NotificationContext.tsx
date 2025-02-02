"use client";
import { createContext, useState, useContext } from "react";

interface Notification {
  id: string;
  type: "error" | "warn" | "info";
  originPage: string;
  message: string;
}

interface NotificationContextType {
  notifications: Notification[];
  count: number;
  addNotification: (notification: Notification) => void;
  deleteNotification: (id: string) => void;
  modalOpen: boolean;
  setModalOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

const NotificationContext = createContext<NotificationContextType>(
  {} as NotificationContextType
);

export const useNotificationContext = () => useContext(NotificationContext);

export const NotificationProvider = ({ children }: any) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [count, setCount] = useState<number>(0);
  const [modalOpen, setModalOpen] = useState<boolean>(false);

  const addNotification = (notification: Notification) => {
    setNotifications([...notifications, notification]);
    setCount(count + 1);
  };

  const deleteNotification = (id: string) => {
    const updatedNotifications = notifications.filter(
      (notif: Notification) => notif.id != id
    );
    setNotifications(updatedNotifications);
    setCount(count - 1);
  };

  const values = {
    notifications,
    addNotification,
    count,
    deleteNotification,
    modalOpen,
    setModalOpen,
  };

  return (
    <NotificationContext.Provider value={values}>
      {children}
    </NotificationContext.Provider>
  );
};
