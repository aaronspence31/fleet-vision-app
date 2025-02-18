import { getDocs, query, collection, orderBy, limit, where } from 'firebase/firestore';
import { FrameBatchType } from './general';
import { db } from "@/lib/firebase";

export const getData = async (collectionName: string) => {
    const querySnapshot = await getDocs(collection(db, collectionName));
    if (querySnapshot.empty) {
        console.log('No matching documents.');
        return [];
    }
    const documents: FrameBatchType[] = querySnapshot.docs.map(doc => (doc.data() as FrameBatchType));
    return documents;
}

// This needs to be updated, it is giving garbage data right now to make the deployment work
export const getDocumentsByRecentSession = async (collectionName: string) => {
  const q = query(collection(db, collectionName), limit(20)); 
  const querySnapshot = await getDocs(q)
  if (querySnapshot.empty) {
      console.log('No matching documents.');
      return [];
  }
  const documents: FrameBatchType[] = querySnapshot.docs.map(doc => (doc.data() as FrameBatchType));
  return documents;
  };