import numpy as np
import matplotlib.pyplot as plt
import math as ma

#---------------- Ce projet a été réalisé par : Roess Celia / Sanchez Mathew / Sekkat Adam -------------

def matrix_initialisation(filepath):
  image_matrix = plt.imread(filepath)
  if len(np.shape(image_matrix)) == 3:  #Image avec Couleur
    lignes, colonnes, canaux = np.shape(image_matrix)  #on récupère le nombre de lignes, de colonnes et de canaux
    #on récupère les plus grand multiples de 8 pour chaque composante
    newLignes = (lignes // 8) * 8
    newColonnes = (colonnes // 8) * 8
    #on tronque l'image à des multiples de 8 en x et y
    image_matrix_tronque = image_matrix[0:newLignes, 0:newColonnes, 0:3]
    
  elif len(np.shape(image_matrix)) == 2:  #Image en Noir et Blanc
    #on récupère le nombre de lignes, de colonnes et de canaux dans notre matrice lignes, colonnes et on met canaux = 1
    lignes, colonnes = np.shape(image_matrix)
    canaux = 1
    #on récupère les plus grands multiples de 8 pour chaque composante
    newLignes = (lignes // 8) * 8
    newColonnes = (colonnes // 8) * 8
    image_matrix_tronque = image_matrix[0:newLignes, 0:newColonnes]
  
  if filepath[-3:] == 'png':  #Image en format png
    #Dans le format png, les intensités sont initialement comprises entre 0 et 1.
    #on change à mettre les intensités entre 0 et 255
    new_matrix = image_matrix_tronque * 255 
    #pour passer de valeur entre 0 et 1 à des valeurs entre 0 et 255
    new_matrix = new_matrix.astype('uint8')  #pour passer de float a int
    centred_matrix = new_matrix - 128  #pour centrer les valeurs
    centred_matrix = centred_matrix.astype('int8')  #pour avoir des int entre -127 et 128
    return centred_matrix
    
  if (filepath[-4:] == 'jpeg' or filepath[-3:] == 'jpg'):  # Image en format jpeg ou jpg
    #Dans le format png, les intensités sont initialement comprises entre 0 et 255.
    centred_matrix = image_matrix_tronque - 128  #pour centrer les valeurs
    centred_matrix = centred_matrix.astype('int8')  #pour avoir des int entre -127 et 128
    return centred_matrix
    

def PMatrix():
  #On initialise notre matrice de passage P
  P = np.zeros((8, 8))  #On crée une matrice nulle de taille 8x8
  c0 = 1 / (np.sqrt(2))  #c0 donné c0=1/sqrt(2)
  for i in range(1, 8):
    for j in range(8):
      P[0, j] = 0.5 * c0  #pour i=0 P(0,j)= 0.5*c0
      P[i, j] = 1 / 2 * np.cos(
          ((2 * j + 1) * i * ma.pi) / 16)  #pour i>0 Ci = 1
  return P


#On définit la matrice de quantification
Q = np.array([[16, 11, 10, 16, 24, 40, 51,
               61], [12, 12, 13, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])


def decomPMatrix(M, Tronque_lignes, Tronque_Colonnes):
  """
  Pour la fonction decomPMatrix : on parcours la matrice par bloc de 8x8 en colonnes et chaque bloc parcouru sera ajouté au tableau : decoMatrix
  Resultat : decoMatrix tableau contenant tout les blocs 8x8 qui compose notre image
  """
  decoMatrix = []
  for i in range(0, Tronque_lignes, 8):
    for j in range(0, Tronque_Colonnes, 8):
      decoMatrix.append(M[i:i + 8, j:j + 8])
  return decoMatrix


def compression_color(tab, Tronque_lignes, Tronque_Colonnes):
  """
  Pour la fonction Compression : Pour chaque Bloc 8x8 :
  -> On applique le changement de Base D = PMP' 
  -> Ensuite on divise terme à terme par la matrice de Quantification D./Q
  -> et pour chaque bloc auquel on a appliqué les étapes précédentes on le rajoute dans compressed_tab
  -> on calcule le taux de compression

  Resultats : - On a un tableau qui contient tous les blocs 8x8 compressés
              - On affiche le taux de compression

  """
  P = PMatrix()  # on récupère notre matrice P
  Pprime = np.transpose(P)  # on obtient P' : transposée de P
  compressed_tab = []  # Tableau dans lequel on va stocker nos blocs 8x8 compressés
  nb_nonNuls = 0  # compteur pour les éléments non nuls
  for i in range(3):  #On lis les couleurs i=0->rouge ; i=1->vert ; i=2->bleu
    for M in tab:
      Temp = np.matmul(P, M[:, :, i])
      D = np.matmul(Temp, Pprime)  #Appliquer le changement de base
      D = np.divide(D, Q)  #diviser terme à terme par la matrice de quantification D./Q
      D = np.trunc(D)  #on cherche à obtenir la partie entiere
      compressed_tab.append(D)
      nb_nonNuls = nb_nonNuls + np.count_nonzero(D[i])  #on calcule pour chaque itération le nombre d'éléments non nuls
  taux_compression=(1-nb_nonNuls/(Tronque_lignes*Tronque_Colonnes*3))*100  #on calcule le taux de compression
  print(f"Taux de Compression : {taux_compression}")
  return compressed_tab


def compression_bw(tab, Tronque_lignes, Tronque_Colonnes):
  """
  Pour la fonction Compression : Pour chaque Bloc 8x8 :
  -> On applique le changement de Base D = PMP' 
  -> Ensuite on divise terme à terme par la matrice de Quantification D./Q
  -> et pour chaque bloc auquel on a appliqué les étapes précédentes on le rajoute dans compressed_tab
  -> on calcule le taux de compression

  Resultats : - On a un tableau qui contient tous les blocs 8x8 compressés
              - On affiche le taux de compression

  """
  P = PMatrix()  # on récupère notre Matrice P
  Pprime = np.transpose(P)  # on obtient P' : transposée de P
  compressed_tab = []  # Tableau dans lequel on va stocker nos blocs 8x8 compressés
  nb_nonNuls = 0  # Compteur pour les éléments non nuls
  for M in tab:
    Temp = np.matmul(P, M)
    D = np.matmul(Temp, Pprime)  #Appliquer le changement de base
    D = np.divide(
        D, Q)  #Appliquer la matrice de quantification terme à terme D./Q
    D = np.trunc(D)  #pour obtenir la partie entiere
    compressed_tab.append(D)
    nb_nonNuls = nb_nonNuls + np.count_nonzero(
        D)  #calcule pour chaque iteration le nombre d'element non nuls
  taux_compression=(1-nb_nonNuls/(Tronque_lignes*Tronque_Colonnes*3))*100  #calcule du taux de Compression
  print(f"Taux de Compression : {taux_compression}")
  return compressed_tab


def decompression(tab):
  """
    Pour la fonction decompression : Pour chaque Bloc 8x8 Compressé:
    -> On le multiple par la matrice de Quantifiaction Q 
    -> Ensuite applique la tranformée Inverse 
    -> et pour chaque bloc auquel on a appliqué les etapes precedentes on le rajoute dans decompressed_tab
    -> on calcule le taux de compression

    Resultats : - On a un tableau qui contient tout les bloc 8x8 decompressés

    """
  P = PMatrix()  # on recupere notre matrice P
  Pprime = np.transpose(P)  # on recupere la transposee de notre matrice P : P'
  decompressed_tab = [
  ]  # Tableau dans lequel on va stocker nos bloc 8x8 Decompressés
  for M in tab:
    DC = M * Q
    DInter = np.matmul(Pprime, DC)
    D = np.matmul(DInter, P)
    decompressed_tab.append(D)
  return decompressed_tab


def recompMatrix_color(tab, Tronque_lignes, Tronque_Colonnes):
  """
    Fonction qui a partir de nos tableau composé de bloc 8x8 recompose une matrice de la meme dimension que la matrice de base tronqué
    """
  recomposed_matrix = np.zeros((
      Tronque_lignes, Tronque_Colonnes,
      3))  #on cree une matrice 3d aux meme dimension que notre matrice tronqué
  recomposed_matrix_red = np.zeros(
      (Tronque_lignes,
       Tronque_Colonnes))  # on crée une matrice 2D pour le canal Rouge
  recomposed_matrix_green = np.zeros(
      (Tronque_lignes,
       Tronque_Colonnes))  # on crée une matrice 2D pour le canal Vert
  recomposed_matrix_blue = np.zeros(
      (Tronque_lignes,
       Tronque_Colonnes))  #on créé une matrice 2D pour le canal Bleu
  #comme dit precedemment notre tab contient tout les bloc 8x8 :
  #    --> Le premier tiers contient : les bloc 8x8 dont le canal est Rouge
  #    --> Le second tiers contient : les bloc 8x8 dont le canal est Vert
  #    --> Le premier tiers contient : les bloc 8x8 dont le canal est Bleu
  n = len(tab)
  tab1 = np.ravel(tab[0:n // 3])
  tab2 = np.ravel(tab[n // 3:2 * n // 3])
  tab3 = np.ravel(tab[2 * n // 3:n])
  k = 0
  # Chaque bloc 8x8 des matrices qu'on a initialisé plus haut on le remplace par son equivalent dans notre tableau de bloc 8x8
  for i in range(0, Tronque_lignes, 8):
    for j in range(0, Tronque_Colonnes, 8):
      recomposed_matrix_red[i:i + 8,
                            j:j + 8] = np.reshape(tab1[64 * k:64 * (k + 1)],
                                                  (8, 8))
      recomposed_matrix_green[i:i + 8,
                              j:j + 8] = np.reshape(tab2[64 * k:64 * (k + 1)],
                                                    (8, 8))
      recomposed_matrix_blue[i:i + 8,
                             j:j + 8] = np.reshape(tab3[64 * k:64 * (k + 1)],
                                                   (8, 8))
      k = k + 1
  recomposed_matrix[:, :, 0] = recomposed_matrix_red
  recomposed_matrix[:, :, 1] = recomposed_matrix_green
  recomposed_matrix[:, :, 2] = recomposed_matrix_blue
  return recomposed_matrix


def recompMatrix_bw(tab, Tronque_lignes, Tronque_Colonnes):
  """
    Fonction qui a partir de nos tableau composé de bloc 8x8 recompose une matrice de la meme dimension que la matrice de base tronqué
    """
  recomposed_matrix = np.zeros(
      (Tronque_lignes, Tronque_Colonnes
       ))  #on cree une matrice 3d aux meme dimension que notre matrice tronqué
  tab1 = np.ravel(tab)
  k = 0
  # Chaque bloc 8x8 des matrices qu'on a initialisé plus haut on le remplace par son equivalent dans notre tableau de bloc 8x8
  for i in range(0, Tronque_lignes, 8):
    for j in range(0, Tronque_Colonnes, 8):
      recomposed_matrix[i:i + 8,
                        j:j + 8] = np.reshape(tab1[64 * k:64 * (k + 1)],
                                              (8, 8))
      k = k + 1
  return recomposed_matrix

def decomposition_hf(matrix,lignes,colonnes):
  decomp = []
  for i in range(0,lignes,8):
      for j in range(0,colonnes,8):
          decomp.append(matrix[i:i+8,j:j+8])
  return decomp

def compression_hf(M,P):
  D = np.zeros((8,8))
  Dmid = np.zeros((8,8))
  Dmid = np.matmul(P,M)
  D = np.matmul(Dmid,np.transpose(P))
  return D

def highfilter(D,threshold):
  """
  fonction de filtrage des hautes fréquences
  i est l'indice des lignes, j est l'indice des colonnes. Si i+j>threshold, le coefficient
  à l'indice (i,j) devient 0
  """
  for i in range(8):
    for j in range(8):
      if i+j>=threshold:
        D[i,j]=0
  return D

def decompression_hf(D,P):
  M = np.zeros((8,8))
  Mmid = np.zeros((8,8))
  Mmid = np.matmul(np.transpose(P),D)
  M = np.matmul(Mmid,P)
  return M

def postprocessing_couleur(mat_init, mat_fin):
  erreur = []
  for k in range(3):
    interm = (np.linalg.norm(mat_init[:,:,k])-np.linalg.norm(mat_fin[:,:,k])) / np.linalg.norm(mat_init[:,:,k])
    erreur.append(interm)
  error = np.mean(erreur) * 100
  return error

def postprocessing_bw(mat_init, mat_fin):
  erreur = (np.linalg.norm(mat_init)-np.linalg.norm(mat_fin)) / np.linalg.norm(mat_init)
  error = erreur * 100
  return error

def num_fichier():
  '''
  On crée une matrice qui permettra à chaque itération d'avoir un nom de fichier différent.
  Par exemple :
    La première fois que l'on effectuera notre programme Python, le fichier créé sera compression1.png
    La deuxième fois, le fichier créé sera compression2.png
    etc...
  '''
  # Nom du fichier pour stocker la valeur
  nom_fichier = "compteur.txt"

  # Essayer d'ouvrir le fichier en mode lecture
  try:
    with open(nom_fichier, 'r') as fichier:
      # Lire la valeur actuelle du compteur
      compteur = int(fichier.read().strip())
  except FileNotFoundError:
    # Si le fichier n'existe pas, initialiser le compteur à 0
    compteur = 0

  # Incrémenter le compteur
  compteur += 1

  # Écrire la nouvelle valeur dans le fichier
  with open(nom_fichier, 'w') as fichier:
    fichier.write(str(compteur))

  return compteur

if __name__ == "__main__":
  filepath = input('Entrez le filepath : ')
  M = matrix_initialisation(filepath)
  Tronque_lignes = np.shape(M)[0]
  Tronque_Colonnes = np.shape(M)[1]
  choix = input('Veuillez Choisir le Moyen de Compression : \n 1 : Division par la matrice de quantification \n 2 : Filtrage des hautes fréquences \n')
  if choix == "1" :
    tab = decomPMatrix(M,Tronque_lignes,Tronque_Colonnes)
    if len(np.shape(M)) == 3 :
      comp_color = compression_color(tab,Tronque_lignes,Tronque_Colonnes)
      decomp_color = decompression(comp_color)
      recomp_color = recompMatrix_color(decomp_color,Tronque_lignes,Tronque_Colonnes)
      MatriceDecompressInterm = recomp_color + 128 
      MatriceDecompress = MatriceDecompressInterm / 255
      print(f"Pourcentage d'erreur :  {postprocessing_couleur(M,recomp_color)} %")
      plt.imshow(MatriceDecompress)
      plt.imsave(f"./compression{num_fichier()}.png",np.clip(MatriceDecompress,0,1))
      plt.show()
    else :
      comp_bw = compression_bw(tab,Tronque_lignes,Tronque_Colonnes)
      decomp_bw = decompression(comp_bw)
      recomp_bw = recompMatrix_bw(decomp_bw,Tronque_lignes,Tronque_Colonnes)
      MatriceDecompressInterm = recomp_bw + 128 
      MatriceDecompress = MatriceDecompressInterm / 255 
      print(f"Pourcentage d'erreur :  {postprocessing_bw(M,recomp_bw)} %")
      plt.imshow(MatriceDecompress,cmap='gray')
      plt.imsave(f"./compression{num_fichier()}.png",np.clip(MatriceDecompress,0,1),cmap='gray')
      plt.show()
  else :
    if len(np.shape(M)) == 3 :
    #Pour chacune des couleurs on décompose la matrice initiale en un tableau de matrices 8*8
      P = PMatrix() 
      treshold = input('Veuillez choisir la Frequence de Coupure \n')
      red = decomposition_hf(M[:,:,0], Tronque_lignes, Tronque_Colonnes)
      green = decomposition_hf(M[:,:,1], Tronque_lignes, Tronque_Colonnes)
      blue = decomposition_hf(M[:,:,2], Tronque_lignes, Tronque_Colonnes)
      non_zeros=0
      """
      pour chaque matrice D dans chaque tableau de couleur on applique la DCT (D = PMP')
      puis on applique le filtrage des hautes fréquences
      """
      for i in range(len(red)):
        red[i]=highfilter(compression_hf(red[i],P),int(treshold))
        green[i]=highfilter(compression_hf(green[i],P),int(treshold))
        blue[i]=highfilter(compression_hf(blue[i],P),int(treshold))
        non_zeros += np.count_nonzero(red[i]) + np.count_nonzero(green[i]) + np.count_nonzero(blue[i])
      taux_compression_hf = (1 - non_zeros/(Tronque_lignes*Tronque_Colonnes*3))*100
      print(f"Taux de Compression pour Haute Frequence: {taux_compression_hf}")
      #décompression de chaque matrice dans chaque tableau de couleur
      for i in range(len(red)):
        red[i]=decompression_hf(red[i],P)
        green[i]=decompression_hf(green[i],P)
        blue[i]=decompression_hf(blue[i],P)
      #création de la nouvelle matrice de l'image
      #initialisation de la nouvelle matrice d'image
      recomposed_matrix_hf = np.zeros((Tronque_lignes,Tronque_Colonnes,3))
      recomposed_matrix_red_hf = np.zeros((Tronque_lignes,Tronque_Colonnes))
      recomposed_matrix_green_hf = np.zeros((Tronque_lignes,Tronque_Colonnes))
      recomposed_matrix_blue_hf = np.zeros((Tronque_lignes,Tronque_Colonnes))
      #on transforme les tableaux de matrices en vecteurs, pour chaque couleur
      n_hf = len(red)
      red1 = np.ravel(red[0:n_hf])#rouges
      green1 = np.ravel(green[0:n_hf])#verts
      blue1 = np.ravel(blue[0:n_hf])#bleus
      #on range les éléments du vecteur en matrices 8x8 dans les canaux (couleurs) de la nouvelle matrice d'image
      k=0
      for i in range(0,Tronque_lignes,8):
        for j in range(0,Tronque_Colonnes,8):
          recomposed_matrix_red_hf[i:i+8,j:j+8] = np.reshape(red1[64*k : 64*(k+1)],(8,8))
          recomposed_matrix_green_hf[i:i+8,j:j+8] = np.reshape(green1[64*k : 64*(k+1)],(8,8))
          recomposed_matrix_blue_hf[i:i+8,j:j+8] = np.reshape(blue1[64*k : 64*(k+1)],(8,8))
          k = k+1
      #1er canal : matrices du rouge
      #2e canal : matrices du vert
      #3e canal : matrices du bleu
      recomposed_matrix_hf[:,:,0] = recomposed_matrix_red_hf
      recomposed_matrix_hf[:,:,1] = recomposed_matrix_green_hf
      recomposed_matrix_hf[:,:,2] = recomposed_matrix_blue_hf
      print(f"Pourcentage d'erreur :  {postprocessing_couleur(M,recomposed_matrix_hf)} %")
      #on transforme les valeurs pour les avoir entre 0 et 1
      recomposed_matrix_hf=np.trunc(recomposed_matrix_hf)
      recom255 = recomposed_matrix_hf + 128
      recom01 = recom255 / 255
      #on affiche la nouvelle image à partir de la nouvelle matrice
      plt.imshow(recom01)
      plt.imsave(f"./compression{num_fichier()}.png",np.clip(recom01,0,1))
      plt.show()
      #np.clip permet d'avoir toutes les valeurs entre 0 et 1 (il y en a qui dépassent)
    else :
      P = PMatrix() 
      treshold = input('Veuillez choisir la Frequence de Coupure \n')
      M_decomp = decomposition_hf(M, Tronque_lignes, Tronque_Colonnes)
      non_zeros=0
      for i in range(len(M_decomp)):
        M_decomp[i]=highfilter(compression_hf(M_decomp[i],P),int(treshold))
        non_zeros += np.count_nonzero(M_decomp[i])
      taux_compression_hf = (1 - non_zeros/(Tronque_lignes*Tronque_Colonnes*3))*100
      print(f"Taux de Compression pour Haute Frequence: {taux_compression_hf}")
      #décompression de chaque matrice dans chaque tableau de couleur
      for i in range(len(M_decomp)):
        M_decomp[i]=decompression_hf(M_decomp[i],P)
      #création de la nouvelle matrice de l'image
      #initialisation de la nouvelle matrice d'image
      recomposed_matrix_hf = np.zeros((Tronque_lignes,Tronque_Colonnes))
      #on transforme les tableaux de matrices en vecteurs, pour chaque couleur
      n_hf = len(M_decomp)
      M_decomp1 = np.ravel(M_decomp[0:n_hf])
      #on range les éléments du vecteur en matrices 8x8 dans les canaux (couleurs) de la nouvelle matrice d'image
      k=0
      for i in range(0,Tronque_lignes,8):
        for j in range(0,Tronque_Colonnes,8):
          recomposed_matrix_hf[i:i+8,j:j+8] = np.reshape(M_decomp1[64*k : 64*(k+1)],(8,8))
          k = k+1
      print(f"Pourcentage d'erreur :  {postprocessing_bw(M,recomposed_matrix_hf)} %")
      #on transforme les valeurs pour les avoir entre 0 et 1
      recomposed_matrix_hf=np.trunc(recomposed_matrix_hf)
      recom255 = recomposed_matrix_hf + 128
      recom01 = recom255 / 255
      #on affiche la nouvelle image à partir de la nouvelle matrice
      plt.imshow(recom01,cmap='gray')
      #np.clip permet d'avoir toutes les valeurs entre 0 et 1 (il y en a qui dépassent)
      plt.imsave(f"./compression{num_fichier()}.png",np.clip(recom01,0,1),cmap='gray')
      plt.show()
      
      