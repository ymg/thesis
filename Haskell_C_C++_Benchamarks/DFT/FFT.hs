
import           Data.Complex
import           Samples
import           Strategies
import           System.Console.GetOpt
import           System.Environment


------------------
-- twiddle factors
tw :: Int -> Int -> Complex Float
tw n k = cis (-2 * pi * fromIntegral k / fromIntegral n)

-- Discrete Fourier Transform -- O(n^2)
dft :: [Complex Float] -> [Complex Float]
-- dft xs  = [parFold rdeepseq (\x y -> sum x y) 0 $ parzipwith (\j k -> xs!!j * tw n (j*k)) [0..n'] [0..n']] --[sum [ xs!!j * tw n (j*k) | j <- [0..n']] | k <- [0..n']]
dft xs = parMap rdeepseq sum $ parMap rpar (\x -> map (\j -> xs!!j * tw n (j*x)) [0..n'] ) [0..n']
         where
            n  = length xs
            n' = n - 1

-- Fast Fourier Transform

-- In case you are wondering, this is the Decimation in Frequency (DIF)
-- radix 2 Cooley-Tukey FFT

fft :: [Complex Float] -> [Complex Float]
fft [a] = [a]
fft as  = dC rpar split threshold combine worker as  
      where
        worker        = (\t ->  [unpackT $ bflyS t])
        combine       = concat
        threshold     = (\x ->  length x < 1000)
        
        split l       = [ls, rs]
          where (ls, rs) = splitAt (length l `div` 2) l
 
        unpackT (a,b) = interleave (fft a) (fft b) 


interleave [] bs = bs
interleave (a:as) bs = a : interleave bs as

bflyS :: [Complex Float] -> ([Complex Float], [Complex Float])
bflyS as = (los,rts)
  where
    (ls,rs) = halve as
    los = zipWith (+) ls rs
    ros = zipWith (-) ls rs
    rts = zipWith (*) ros [tw (length as) i | i <- [0..(length ros) - 1]]


-- split the input into two halves
halve as = splitAt n' as
  where
    n' = div (length as + 1) 2

-- the main function
-- uses Samples.samples to generate some sample data
--   samples :: Int -> Int -> [Complex Float]

defsize = 1000 -- change this to get larger samples
defseed = 1

main = do args <- getArgs
          let arglen = length args
          let n      = argval args 0 defsize
          let seed   = argval args 1 defseed
          let fun    = if arglen > 2 && args !! 2 == "dft" then dft else fft
          print (sum (fun (samples seed n)))

argval args n def = if length args > n then
                       read (args !! n)
                    else def
