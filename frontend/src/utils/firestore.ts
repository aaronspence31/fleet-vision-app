import { getDocs, query, collection, orderBy, limit, where } from 'firebase/firestore';
import { FrameBatchType } from './general';
import { db } from "@/lib/firebase/firebase";

export const getData = async (collectionName: string) => {
    const querySnapshot = await getDocs(collection(db, collectionName));
    if (querySnapshot.empty) {
        console.log('No matching documents.');
        return [];
    }
    const documents: FrameBatchType[] = querySnapshot.docs.map(doc => (doc.data() as FrameBatchType));
    return documents;
}

export const getDocumentsByRecentSession = async (collection_name: string) => {
    try {
      // Step 1: Get the document with the most recent timestamp_start
      const q = query(
        collection(db, collection_name),
        orderBy('timestamp_start', 'desc'),
        limit(1) // Get only the most recent document
      );
      const snapshot = await getDocs(q);
  
      if (!snapshot.empty) {
        // Step 2: Retrieve the session_id from the most recent document
        const mostRecentDoc = snapshot.docs[0];
        const sessionId = mostRecentDoc.data().session_id;
        console.log("Most recent doc: ", mostRecentDoc);
  
        // Step 3: Query all documents with the same session_id
        const sessionQuery = query(
          collection(db, collection_name),
          where('session_id', '==', sessionId)
        );
        const resultSnapshot = await getDocs(sessionQuery);
  
        if (!resultSnapshot.empty) {
          // Return all documents with the matching session_id
          const documents = resultSnapshot.docs.map(doc => (doc.data() as FrameBatchType));
          return documents; // Return the array of documents
        } else {
          console.log('No documents found with the same session_id');
          return []; // Return an empty array if no documents match the session_id
        }
      } else {
        console.log('No documents found');
        return []; // Return an empty array if no document is found with the most recent timestamp_start
      }
    } catch (error) {
      console.error('Error retrieving documents:', error);
      throw error; // Re-throw the error if any occurs
    }
  };
