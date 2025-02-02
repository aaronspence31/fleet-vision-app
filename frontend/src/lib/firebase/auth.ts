  import { auth, db} from "./firebase";
  import { defaultUserObject } from "../types";
  import {
    signInWithPopup,
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    updateProfile,
    sendEmailVerification,
    GoogleAuthProvider
  } from "firebase/auth";
  import {
    query,
    getDocs,
    setDoc,
    collection,
    where,
    doc,
    updateDoc,
    DocumentData,
  } from "firebase/firestore";

  interface FirebaseErrorCode {
    code: string,
    customData: {},
    name: string
}

  const provider = new GoogleAuthProvider();
  
  export const signUpWithEmail = async ( 
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    cid: string,
  ) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password,
      );
      const { user } = userCredential;
      console.log("user: ", user)
      await setDoc(doc(db, "users", user.uid), defaultUserObject);
      await updateProfile(user, {
        displayName: firstName,
        photoURL:
          "https://i.pinimg.com/originals/f6/bc/9a/f6bc9a75409c4db0acf3683bab1fab9c.png",
      });
      const usersRef = doc(db, "users", user.uid);
      await updateDoc(usersRef, {
        firstName: firstName,
        lastName: lastName,
        pfp: "https://i.pinimg.com/originals/f6/bc/9a/f6bc9a75409c4db0acf3683bab1fab9c.png",
        cid: cid,
      });
      // await sendEmailVerification(user);
      return user;
    } catch (error) {
      console.log(error);
      return error;
    }
  };
  
  export const signInWithEmail = async (email: string, password: string) => {
    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email,
        password,
      );
      return userCredential.user;
    } catch (error) {
      console.log(error);
      return error
    }
  };

  export const signInWithGoogle = async () => {
    try {
      const result = await signInWithPopup(auth, provider);
      return result.user;
    } catch (error) {
      console.log(error);
      return error;
    }
  }
  
  export const logout = async () => {
    try {
      await auth.signOut();
    } catch (error) {
      console.log(error);
    }
  };
  